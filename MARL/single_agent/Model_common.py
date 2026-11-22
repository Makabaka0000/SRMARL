import torch as th
from torch import nn
import torch.nn.functional as F
import math

class ActorNetwork(nn.Module):
    """
    A network for actor
    """

    def __init__(self, state_dim, hidden_size, output_size, output_act):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # activation function for the output
        self.output_act = output_act

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        out = self.output_act(self.fc3(out))
        return out


class CriticNetwork(nn.Module):
    """
    A network for critic
    """

    def __init__(self, state_dim, action_dim, hidden_size, output_size=1):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def __call__(self, state, action):
        out = nn.functional.relu(self.fc1(state))
        out = th.cat([out, action], 1)
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out



class CostCriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ActorCriticNetwork(nn.Module):
    """
    An actor-critic network that shared lower-layer representations but
    have distinct output layers
    """

    def __init__(self, state_dim, action_dim, hidden_size,
                 actor_output_act, critic_output_size=1):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear = nn.Linear(hidden_size, action_dim)
        self.critic_linear = nn.Linear(hidden_size, critic_output_size)
        self.actor_output_act = actor_output_act

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        act = self.actor_output_act(self.actor_linear(out))
        val = self.critic_linear(out)
        return act, val


class CVaRCurriculum:
    def __init__(
        self,
        n_agents,
        num_stages,
        alpha_start=0.1,
        alpha_end=0.95,
        cost_limit_start=None,
        cost_limit_end=None,
        lambda_cap_start=0,
        lambda_cap_end=10000000,
        schedule="linear"
    ):
        self.n_agents = n_agents
        self.num_stages = num_stages
        self.stage = 0

        self.alpha_start = alpha_start
        self.alpha_end = alpha_end

        assert cost_limit_start is not None
        assert cost_limit_end is not None
        self.cost_limit_start = th.tensor(cost_limit_start, dtype=th.float32)
        self.cost_limit_end = th.tensor(cost_limit_end, dtype=th.float32)

        self.lambda_cap_start = lambda_cap_start
        self.lambda_cap_end = lambda_cap_end

        self.schedule = schedule
        self._update()

    def step(self):
        self.stage = min(self.stage + 1, self.num_stages - 1)
        self._update()

    def _interp(self, start, end):
        ratio = self.stage / (self.num_stages - 1)
        if self.schedule == "linear":
            return start + ratio * (end - start)
        elif self.schedule == "cosine":
            r = 0.5 * (1 - th.cos(th.tensor(ratio * 3.1415926)))
            return start + r * (end - start)
        else:
            raise NotImplementedError

    def _update(self):
        # ---- alpha ----
        self.alpha = float(self._interp(self.alpha_start, self.alpha_end))

        # ---- cost limit (beta) ----
        beta = self._interp(self.cost_limit_start, self.cost_limit_end)
        self.cost_limit = beta.clone()  # [n_agents]

        # ---- lambda cap ----
        self.lambda_cap = float(self._interp(
            self.lambda_cap_start, self.lambda_cap_end
        ))


class CostDistributionModel(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_dim=256):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Gaussian head
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

        # Tail probability
        self.pi_tail_head = nn.Linear(hidden_dim, 1)

        # Tail parameters (example: GPD with shape ξ and scale β)
        self.tail_param_head = nn.Linear(hidden_dim, 2)

    def forward(self, s, a):
        """
        s: [B, s_dim]
        a: [B, a_dim]
        """
        x = th.cat([s, a], dim=-1)
        # backbone
        h = self.backbone(x)

        # Gaussian head
        mu = self.mu_head(h).squeeze(-1)
        log_sigma = self.log_sigma_head(h).clamp(-20, 2)  # 防止 exp overflow
        sigma = th.exp(log_sigma) + 1e-6

        # Tail probability
        pi_tail = th.sigmoid(self.pi_tail_head(h)).clamp(1e-6, 1-1e-6)

        # Tail parameters (GPD)
        tail_raw = self.tail_param_head(h)
        xi = tail_raw[:, 0].clamp(-0.5, 0.5)
        beta = F.softplus(tail_raw[:, 1]) + 1e-6
        tail_param = th.stack([xi, beta], dim=-1)

        return mu, sigma, pi_tail, tail_param

    def compute_log_prob(self, C, mu, sigma, pi_tail, tail_param):
        """
        Piecewise Gaussian-GPD log-likelihood.
        Gaussian body is valid for 0 <= C < u (interval-truncated, costs are non-negative).
        GPD tail is valid for C >= u.
        """

        eps = 1e-6

        C = C.view(-1, 1)
        mu = mu.view(-1, 1)
        sigma = sigma.view(-1, 1).clamp(min=eps)
        pi_tail = pi_tail.view(-1, 1).clamp(min=eps, max=1.0 - eps)

        xi = tail_param[:, 0].view(-1, 1).clamp(min=-0.99, max=1.0)
        beta = tail_param[:, 1].view(-1, 1).clamp(min=1e-3)

        # empirical threshold: u = Q_0.9(C)
        u = th.quantile(C.detach().view(-1), 0.9)

        # -------------------------------
        # Truncated Gaussian log-likelihood
        # valid for 0 <= C < u
        # -------------------------------
        log_norm_const = 0.5 * th.log(th.tensor(2.0 * th.pi, device=C.device))

        ll_gauss = -0.5 * ((C - mu) / sigma) ** 2 - th.log(sigma) - log_norm_const

        # normalization for the interval-truncated region 0 <= C < u
        # (cost returns are non-negative, so the body is truncated at 0 below)
        # Computed in float64: when [0, u] lies deep in a Gaussian tail,
        # float32 erf saturates and Phi_hi - Phi_lo collapses to 0.
        # Phi(x) = 0.5 * erfc(-x / sqrt(2)) stays accurate in the tails.
        sqrt2 = math.sqrt(2.0)
        mu_d = mu.double()
        sigma_d = sigma.double()
        u_d = u.double()
        a_hi = (u_d - mu_d) / sigma_d
        a_lo = -mu_d / sigma_d
        Phi_hi = 0.5 * th.erfc(-a_hi / sqrt2)
        Phi_lo = 0.5 * th.erfc(-a_lo / sqrt2)
        norm_const = (Phi_hi - Phi_lo).clamp(min=1e-12)

        ll_gauss_tr = ll_gauss - th.log(norm_const).to(C.dtype)

        # -------------------------------
        # GPD log-likelihood
        # valid for C >= u
        # -------------------------------
        z = C - u
        z_tail = z.clamp(min=0.0)

        xi_safe = th.where(
            th.abs(xi) < 1e-4,
            th.full_like(xi, 1e-4),
            xi
        )

        support = 1.0 + xi_safe * z_tail / beta
        valid_support = support > eps

        ll_tail_gpd = -th.log(beta) - (1.0 / xi_safe + 1.0) * th.log(
            support.clamp(min=eps)
        )

        # xi -> 0 case: GPD becomes exponential
        ll_tail_exp = -th.log(beta) - z_tail / beta

        ll_tail = th.where(
            th.abs(xi) < 1e-4,
            ll_tail_exp,
            ll_tail_gpd
        )

        # invalid GPD support
        neg_inf = th.full_like(ll_tail, -1e10)
        ll_tail = th.where(valid_support, ll_tail, neg_inf)

        # -------------------------------
        # Piecewise log-likelihood
        # -------------------------------
        body_mask = C < u

        log_prob_body = th.log(1.0 - pi_tail) + ll_gauss_tr
        log_prob_tail = th.log(pi_tail) + ll_tail

        log_prob = th.where(body_mask, log_prob_body, log_prob_tail)

        return log_prob.squeeze(-1)