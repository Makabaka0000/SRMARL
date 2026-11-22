import torch as th
from torch import nn
import os
import math
import configparser

# config_dir = 'configs/configs_ppo.ini'

base_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(base_dir, 'configs', 'configs_ppo.ini')
config = configparser.ConfigParser()
config.read(config_dir)
torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')
th.manual_seed(torch_seed)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True

from torch.optim import Adam, RMSprop

import numpy as np
import os, logging
from copy import deepcopy
from single_agent.Memory_common_CVaR import OnPolicyReplayMemoryCVaR
from single_agent.Model_common import ActorNetwork, CriticNetwork,CostDistributionModel,CVaRCurriculum
from common.utils import index_to_one_hot, to_tensor_var, VideoRecorder
from torch.nn import functional as F
class MAPPOCVaRG:
    """
    An multi-agent learned with PPO
    reference: https://github.com/ChenglongChen/pytorch-DRL
    """
    def __init__(self, env, state_dim, action_dim,total_training_episodes,writer=None,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=1, target_tau=1.,
                 target_update_steps=5, clip_param=0.2,
                 reward_gamma=0.99, reward_scale=20,
                 actor_hidden_size=128, critic_hidden_size=128,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.0001, critic_lr=0.0001, test_seeds=0,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 traffic_density=1, reward_type="regionalR",use_cuda=True):

        assert traffic_density in [1, 2, 3]
        assert reward_type in ["regionalR", "global_R"]
        self.reward_type = reward_type
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.writer=writer
        self.env_state, self.action_mask = self.env.reset()
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.test_seeds = test_seeds
        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.traffic_density = traffic_density
        self.memory = OnPolicyReplayMemoryCVaR(memory_capacity)
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.actor_output_act = actor_output_act
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.cost_critic_lr = 1e-4
        self.optimizer_type = optimizer_type
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        self.use_cuda = use_cuda and th.cuda.is_available()
        self.roll_out_n_steps = roll_out_n_steps
        self.target_tau = target_tau
        self.target_update_steps = target_update_steps
        self.clip_param = clip_param
        self.n_agents = len(self.env.controlled_vehicles)
        self.critic_dim=len(self.env.controlled_vehicles)*5

        self.lagrangian_multiplier = np.zeros(self.n_agents, dtype=np.float32)
        self.lagrangian_lr = 0.01         # λ 的学习率

        self.cost_threshold = 5       # cost 上限，例如平均碰撞率 ≤ 0.1
        self.lambda_max=1000.0              # λ 的最大值

        self.cost_limit=np.zeros(self.n_agents)+self.cost_threshold

        self.cost_limit_start=np.zeros(self.n_agents)+3
        self.cost_limit_end=np.zeros(self.n_agents)+1

        self.discover_threshold=0.05
        self.observer_num=5
        self.feature_dim=5
        # 学习率调整
        self.total_training_episodes =total_training_episodes
        self.lr_update_interval=self.total_training_episodes /10
        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act)
        self.critic = CriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1)
        self.actor_target=deepcopy(self.actor)
        self.distribution_model=CostDistributionModel(self.state_dim,action_dim,critic_hidden_size)

        self.curriculum = CVaRCurriculum(
            n_agents=self.n_agents,
            num_stages=10,
            alpha_start=0.6,
            alpha_end=0.95,
            cost_limit_start=self.cost_limit,
            cost_limit_end=self.cost_limit * 0.3,
            lambda_cap_start=10,
            lambda_cap_end=1000,
            schedule="linear"
        )
        # to ensure target network and learning network has the same weights
        self.critic_target = deepcopy(self.critic)

        if self.optimizer_type == "adam":
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.distribution_model_optimizer = Adam(self.distribution_model.parameters(), lr=self.critic_lr)

        elif self.optimizer_type == "rmsprop":
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.distribution_model_optimizer = RMSprop(self.distribution_model.parameters(), lr=self.critic_lr)

        self.actor_scheduler = []

        for agent_id in range(self.n_agents):
            scheduler = th.optim.lr_scheduler.LambdaLR(
                self.actor_optimizer,
                lr_lambda=lambda step: 1 - step / self.total_training_episodes
            )
            self.actor_scheduler.append(scheduler)

        self.episode_rewards = [0]
        self.average_speed = [0]
        self.epoch_steps = [0]

    # agent interact with the environment to collect experience
    def interact(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state, _ = self.env.reset()
            self.n_steps = 0
        states = []
        actions = []
        rewards = []
        costs=[]
        done = True
        average_speed = 0
        # take n steps
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action = self.exploration_action(self.env_state, self.n_agents)
            next_state, global_reward, done, info = self.env.step(tuple(action))
            actions.append([index_to_one_hot(a, self.action_dim) for a in action])
            self.episode_rewards[-1] += global_reward
            self.epoch_steps[-1] += 1
            if self.reward_type == "regionalR":
                reward = info["regional_rewards"]
            elif self.reward_type == "global_R":
                reward = [global_reward] * self.n_agents
            rewards.append(reward)
            costs.append(info["agents_costs"])
            average_speed += info["average_speed"]
            final_state = next_state
            self.env_state = next_state
            self.n_steps += 1
            if done:
                self.env_state, _ = self.env.reset()
                break

        # discount reward
        if done:
            final_value = [0.0] * self.n_agents
            self.n_episodes += 1
            self.episode_done = True
            self.episode_rewards.append(0)
            self.average_speed[-1] = average_speed / self.epoch_steps[-1]
            self.average_speed.append(0)
            self.epoch_steps.append(0)
        else:
            self.episode_done = False
            final_action = self.action(final_state)
            final_value = self.value(final_state, final_action)

        if self.reward_scale > 0:
            rewards = np.array(rewards) / self.reward_scale
            costs = np.array(costs) / self.reward_scale

        for agent_id in range(self.n_agents):
            rewards[:, agent_id] = self._discount_reward(rewards[:, agent_id], final_value[agent_id])

        rewards = rewards.tolist()
        costs = costs.tolist()
        # --- push to memory ---
        self.memory.push(states, actions, rewards,costs)

    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        rewards_var = to_tensor_var(batch.reward_returns, self.use_cuda).view(-1, self.n_agents, 1)
        costs = to_tensor_var(batch.cost, self.use_cuda).view(-1, self.n_agents, 1)


        ##计算长期代价cost_return
        T, N, _ = costs.shape
        cost_returns = th.zeros_like(costs)
        running = th.zeros((N, 1), device=costs.device)
        for t in reversed(range(T)):
            running = costs[t] + self.reward_gamma * running
            cost_returns[t] = running
        #训练Z_c
        T, N, S = states_var.shape
        A = actions_var.shape[-1]
        s_i = states_var.view(T * N, S)        # [TN, S]
        a_i = actions_var.view(T * N, A)        # [TN, A]
        C_i = cost_returns.view(T * N, 1) 
        # empirical 90th percentile threshold
        u_g = th.quantile(C_i.detach().view(-1), 0.9).detach()

        # 1️⃣ forward 得到分布参数
        mu, sigma, pi_tail, tail_param = self.distribution_model(s_i, a_i)
        # 2️⃣ 用安全方法计算混合分布 log-likelihood
        log_prob = self.distribution_model.compute_log_prob(C_i, mu, sigma, pi_tail, tail_param)
        # 3️⃣ 计算分布 critic loss
        distribute_loss = -log_prob.mean()
        if self.writer is not None:
            self.writer.add_scalar(f'distribute_loss', distribute_loss, self.n_episodes)
            self.writer.flush()
        # 4️⃣ backward + 梯度裁剪 + step
        self.distribution_model_optimizer.zero_grad()
        distribute_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.distribution_model.parameters(), self.max_grad_norm)
        self.distribution_model_optimizer.step()
        # update actor network
        M =states_var.shape[0]
        alpha = self.curriculum.alpha
        cost_limit = self.curriculum.cost_limit
        lambda_cap = self.curriculum.lambda_cap

        for agent_id in range(self.n_agents):
            # update actor network
            self.actor_optimizer.zero_grad()
            values = self.critic_target(states_var[:, agent_id, :], actions_var[:, agent_id, :]).detach()
            advantages = rewards_var[:, agent_id, :] - values
            action_log_probs = self.actor(states_var[:, agent_id, :])
            action_log_probs = th.sum(action_log_probs * actions_var[:, agent_id, :], 1)
            old_action_log_probs = self.actor_target(states_var[:, agent_id, :]).detach()
            old_action_log_probs = th.sum(old_action_log_probs * actions_var[:, agent_id, :], 1)
            ratio = th.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * advantages
            surr2 = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            # PPO's pessimistic surrogate (L^CLIP)
            actor_loss = -th.mean(th.min(surr1, surr2))
            with th.no_grad():
                mu_all = mu.view(T, N, -1)
                sigma_all = sigma.view(T, N, -1)
                pi_tail_all = pi_tail.view(T, N, -1)
                tail_param_all = tail_param.view(T, N, -1)

                mu_agent = mu_all[:, agent_id, :].squeeze(-1)
                sigma_agent = sigma_all[:, agent_id, :].squeeze(-1)
                pi_tail_agent = pi_tail_all[:, agent_id, :].squeeze(-1)
                tail_param_agent = tail_param_all[:, agent_id, :]
                cvar = self.compute_cvar_from_distribution(mu_agent,sigma_agent,pi_tail_agent,tail_param_agent,alpha=alpha,u_g=u_g,K=64)
            cvar = cvar.detach()
            cost_surr1 = ratio * cvar
            cost_surr2 = th.clamp(ratio,1.0 - self.clip_param,1.0 + self.clip_param) * cvar
            cost_loss = th.mean(th.max(cost_surr1, cost_surr2))
            lambda_var = self.lagrangian_multiplier[agent_id]
            object_loss = actor_loss + lambda_var * cost_loss
            object_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            self.lagrangian_multiplier[agent_id] += (self.lagrangian_lr *(cvar.mean().detach()- cost_limit[agent_id]))
            # >>> CL-ADD: lambda cap (continuation control)
            self.lagrangian_multiplier[agent_id] = max(0.0,min(self.lagrangian_multiplier[agent_id], lambda_cap))

            if self.writer is not None:
                self.writer.add_scalar(f'lambda/agent_{agent_id}', self.lagrangian_multiplier[agent_id], self.n_episodes)
                self.writer.add_scalar(f'cvar/agent_{agent_id}', cvar.mean(), self.n_episodes)
                self.writer.flush()

            # update critic network
            self.critic_optimizer.zero_grad()
            target_values = rewards_var[:, agent_id, :]
            values = self.critic(states_var[:, agent_id, :], actions_var[:, agent_id, :])
            if self.critic_loss == "huber":
                critic_loss = nn.functional.smooth_l1_loss(values, target_values)
            else:
                critic_loss = nn.MSELoss()(values, target_values)
            critic_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

           
        if self.n_episodes % self.lr_update_interval == 0:##学习率衰减
            for scheduler in self.actor_scheduler:
                scheduler.step()
            self.curriculum.step()

        # update actor target network and critic target network
        if self.n_episodes % self.target_update_steps == 0 and self.n_episodes > 0:
            self._soft_update_target(self.actor_target, self.actor)
            self._soft_update_target(self.critic_target, self.critic)

                
    # predict softmax action based on state
    def _softmax_action(self, state, n_agents):
        state_var = to_tensor_var([state,1], self.use_cuda)

        softmax_action = []
        for agent_id in range(n_agents):
            softmax_action_var = th.exp(self.actor(state_var[:, agent_id, :]))

            if self.use_cuda:
                softmax_action.append(softmax_action_var.data.cpu().numpy()[0])
            else:
                softmax_action.append(softmax_action_var.data.numpy()[0])
        return softmax_action

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state, n_agents):
        softmax_actions = self._softmax_action(state, n_agents)
        actions = []
        for pi in softmax_actions:
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
        return actions

    def action(self, state, n_agents):
        softmax_actions= self._softmax_action_for_interact(state, n_agents)
        actions = []
        for pi in softmax_actions:
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
        return actions

    def _softmax_action_for_interact(self, state, n_agents):
        state_var = to_tensor_var([state,1], self.use_cuda)
        softmax_action = []
        for agent_id in range(n_agents):
            agent_input=self.delay_agent_input(agent_id, state_var)
            softmax_action_var = th.exp(self.actor(agent_input))
            if self.use_cuda:
                softmax_action.append(softmax_action_var.data.cpu().numpy()[0])
            else:
                softmax_action.append(softmax_action_var.data.numpy()[0])
        return softmax_action

    def  delay_agent_input(self,agent_id, states_var):
        local_state = states_var[agent_id, :]  # [batch, state_dim]
        obs = local_state.reshape(self.observer_num, self.feature_dim)
        # 创建一个输出容器
        compensated_obs = obs
        for j in range(1,self.observer_num):
            state = obs[j, :]  # 当前观测状态
            if np.random.rand() > self.discover_threshold:#在决策时间收到了消息
                compensated_obs[j,:]=state
            else:
                # predicted_state =state
                predicted_state =th.zeros(self.feature_dim)
                compensated_obs[j, :] = predicted_state
        compensated_obs = compensated_obs[:, [0, 1, 2, 3, 4]]
        return compensated_obs.reshape(1,-1)

    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action = index_to_one_hot(action, self.action_dim)
        action_var = to_tensor_var([action], self.use_cuda)

        values = [0] * self.n_agents
        for agent_id in range(self.n_agents):
            value_var = self.critic(state_var[:, agent_id, :], action_var[:, agent_id, :])

            if self.use_cuda:
                values[agent_id] = value_var.data.cpu().numpy()[0]
            else:
                values[agent_id] = value_var.data.numpy()[0]
        return values

    # evaluation the learned agent
    def evaluation(self, env, output_dir, eval_episodes=1, is_train=True):
        rewards = []
        costs=[]
        infos = []
        avg_speeds = []
        steps = []
        vehicle_speed = []
        vehicle_position = []
        success_num=0
        crashed=0
        video_recorder = None
        seeds = [int(s) for s in self.test_seeds.split(',')]

        for i in range(eval_episodes):
            avg_speed = 0
            step = 0
            rewards_i = []
            infos_i = []
            costs_i = []
            done = False
            if is_train:
                if self.traffic_density == 1:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 1)
                elif self.traffic_density == 2:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 2)
                elif self.traffic_density == 3:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 3)
            else:
                state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i])

            n_agents = len(env.controlled_vehicles)
            rendered_frame = env.render(mode="rgb_array")
            video_filename = os.path.join(output_dir,
                                          "testing_episode{}".format(self.n_episodes + 1) + '_{}'.format(i) +
                                          '.mp4')
            # Init video recording
            if video_filename is not None:
                print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape,
                                                                      5))
                video_recorder = VideoRecorder(video_filename,
                                               frame_size=rendered_frame.shape, fps=5)
                video_recorder.add_frame(rendered_frame)
            else:
                video_recorder = None

            while not done:
                step += 1
                action = self.action(state, n_agents)
                state, reward, done, info = env.step(action)
                avg_speed += info["average_speed"]
                rendered_frame = env.render(mode="rgb_array")
                if video_recorder is not None:
                    video_recorder.add_frame(rendered_frame)
                costs_i.append(np.mean(info["agents_costs"]))
                rewards_i.append(reward)
                infos_i.append(info)
            if info["agent_crashed"]>=1:
                crashed+=1
            if info["agent_reach"]==n_agents and info["agent_crashed"]==0:
                success_num+=1
            vehicle_speed.append(info["vehicle_speed"])
            vehicle_position.append(info["vehicle_position"])
            rewards.append(rewards_i)
            costs.append(costs_i)
            infos.append(infos_i)
            steps.append(step)
            avg_speeds.append(avg_speed / step)

        if video_recorder is not None:
            video_recorder.release()
        env.close()
        return rewards, costs,(vehicle_speed, vehicle_position), steps, avg_speeds,success_num/len(seeds), crashed/len(seeds)

    # discount roll out rewards
    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    def _compute_gae(self, rewards, values, final_value):
        """
        rewards:      shape [T, 1] or [T]
        values:       V(s_t), shape [T]
        final_value:  V(s_T)
        """
        gae = np.zeros_like(rewards)
        running_gae = 0.0

        values = np.append(values, final_value)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.reward_gamma * values[t + 1] - values[t]
            running_gae = delta + self.reward_gamma * self.gae_lambda * running_gae
            gae[t] = running_gae

        return gae
    
    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    def load(self, model_dir, global_step=None, train_mode=False):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if global_step is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        tokens = file.split('.')[0].split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = file
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)
        if save_file is not None:
            file_path = model_dir + save_file
            checkpoint = th.load(file_path)
            print('Checkpoint loaded: {}'.format(file_path))
            self.actor.load_state_dict(checkpoint['model_state_dict'])
            if train_mode:
                self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.actor.train()
            else:
                self.actor.eval()
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False


    def save(self, model_dir, global_step):
        file_path = model_dir + 'checkpoint-{:d}.pt'.format(global_step)
        th.save({'global_step': global_step,
                 'model_state_dict': self.actor.state_dict(),
                 'optimizer_state_dict': self.actor_optimizer.state_dict()},
                file_path)
        # print(f"Saved checkpoint to {file_path}")

    def exploration_value(self,state):
        state=state[:, :5].reshape(-1)
        joint_state=th.tensor(state, dtype=th.float32)
        value=self.critic_target(joint_state.unsqueeze(0)).detach()
        return value
    
    def compute_gae(self,signal, values, final_value, gamma=0.99, lam=0.95):
        """
        Compute GAE (Generalized Advantage Estimation) and returns for a given signal (reward or cost).

        Args:
            signal (np.array): shape (T, n_agents), reward 或 cost
            values (np.array): shape (T, n_agents), critic 对 signal 的估值
            final_value (np.array or float): shape (n_agents,) 或单个值，如果 done=True
            gamma (float): discount factor
            lam (float): GAE lambda

        Returns:
            advantages (np.array): shape (T, n_agents)
            returns (np.array): shape (T, n_agents)
        """
        signal = np.array(signal, dtype=np.float32)
        values = np.array(values, dtype=np.float32)

        T = len(signal)
        n_agents = values.shape[1] if len(values.shape) > 1 else 1

        # prepare next_values
        next_values = np.zeros_like(values)
        next_values[:-1] = values[1:]
        next_values[-1] = final_value

        advantages = np.zeros_like(signal, dtype=np.float32)
        returns = np.zeros_like(signal, dtype=np.float32)
        gae = np.zeros(n_agents, dtype=np.float32)

        for t in reversed(range(T)):
            delta = signal[t] + gamma * next_values[t] - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns
    
    
    def compute_cvar_from_distribution(self, mu, sigma, pi, tail_param, alpha, u_g, K=64):
        xi = tail_param[:, 0]
        beta = tail_param[:, 1]

        # sample cost return from Gaussian--GPD mixture
        Z = self.sample_mixture(mu, sigma, pi, xi, beta, u_g, K)

        # empirical alpha-quantile, i.e., VaR_alpha
        q = th.quantile(Z, alpha, dim=1, keepdim=True)

        # CVaR_alpha = q + E[(Z-q)^+] / (1-alpha)
        cvar = q + th.relu(Z - q).mean(dim=1, keepdim=True) / (1 - alpha)

        return cvar.squeeze(-1)


    def sample_mixture(self, mu, sigma, pi, xi, beta, u_g, K=64, eps=1e-6):
        """
        Draw K samples from the Gaussian + GPD mixture.

        Body: Gaussian truncated on [0, u_g], sampled by inverse CDF
              (interval-truncated version of Eq. gaussian_sample; cost
              returns are non-negative, so the body is truncated at 0):
              C = mu + sigma * Phi^{-1}( Phi(-mu/sigma)
                                         + z * (Phi((u_g-mu)/sigma) - Phi(-mu/sigma)) ),
              z ~ U(0,1)
        Tail: GPD over the excess above u_g, sampled by inverse CDF.
        Mixture: hard Bernoulli(pi), matching the piecewise likelihood in
                 CostDistributionModel.compute_log_prob.

        Note: samples are used only for CVaR estimation and are detached
        afterwards, so no gradient needs to flow through this function.

        mu, sigma, pi, xi, beta: [B]
        u_g: scalar or [B]
        return: Z [B, K]
        """
        B = mu.shape[0]
        sqrt2 = math.sqrt(2.0)

        # make u_g compatible with [B, 1]
        if not th.is_tensor(u_g):
            u_g = th.tensor(u_g, device=mu.device, dtype=mu.dtype)
        else:
            u_g = u_g.to(device=mu.device, dtype=mu.dtype)

        if u_g.dim() == 0:
            u_g_ = u_g.view(1, 1)
        else:
            u_g_ = u_g.view(B, 1)

        mu64 = mu[:, None].double()
        sigma64 = sigma[:, None].double()
        u64 = u_g_.double()
        b_hi = (u64 - mu64) / sigma64                           # [B, 1]
        b_lo = -mu64 / sigma64                                  # [B, 1]
        Phi_hi = 0.5 * th.erfc(-b_hi / sqrt2)                   # Phi((u_g-mu)/sigma)
        Phi_lo = 0.5 * th.erfc(-b_lo / sqrt2)                   # Phi(-mu/sigma)
        z = th.rand(B, K, device=mu.device, dtype=th.float64)
        p = (Phi_lo + z * (Phi_hi - Phi_lo)).clamp(1e-12, 1.0 - 1e-12)
        # Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1)
        z_g = mu64 + sigma64 * (sqrt2 * th.erfinv(2.0 * p - 1.0))
        # numerical guard: exact-arithmetic samples always lie in [0, u_g]
        z_g = th.minimum(th.maximum(z_g, th.zeros_like(z_g)), u64)
        z_g = z_g.to(mu.dtype)

        # -------------------------------
        # Tail: GPD samples by inverse CDF
        # GPD models the excess over threshold:
        # Y = C - u_g
        # so C = u_g + Y
        # -------------------------------
        u = th.rand(B, K, device=mu.device).clamp(eps, 1 - eps)

        xi_ = xi[:, None]
        beta_ = beta[:, None]

        # excess sample: Y >= 0
        excess = th.where(
            xi_.abs() < 1e-4,
            -beta_ * th.log(u),
            beta_ / xi_ * (u.pow(-xi_) - 1)
        )

        # tail cost sample: C = u_g + Y
        z_t = u_g_ + excess

        # -------------------------------
        # hard Bernoulli mixture (consistent with the piecewise likelihood)
        # m = 1 -> tail sample, m = 0 -> body sample
        # -------------------------------
        pi = pi.clamp(eps, 1.0 - eps)
        m = th.bernoulli(pi[:, None].expand(B, K))

        Z = m * z_t + (1 - m) * z_g

        return Z