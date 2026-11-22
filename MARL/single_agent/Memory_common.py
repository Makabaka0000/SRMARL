import random
from collections import namedtuple


# Experience = namedtuple("Experience",
#                         ("states", "actions", "rewards", "next_states", "dones"))


# class OnPolicyReplayMemory(object):
#     """
#     Replay memory buffer
#     """
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0

#     def _push_one(self, state, action, reward, next_state=None, done=None):
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Experience(state, action, reward, next_state, done)
#         self.position = (self.position + 1) % self.capacity

#     def push(self, states, actions, rewards, next_states=None, dones=None):
#         if isinstance(states, list):
#             if next_states is not None and len(next_states) > 0:
#                 for s,a,r,n_s,d in zip(states, actions, rewards, next_states, dones):
#                     self._push_one(s, a, r, n_s, d)
#             else:
#                 for s,a,r in zip(states, actions, rewards):
#                     self._push_one(s, a, r)
#         else:
#             self._push_one(states, actions, rewards, next_states, dones)

#     def sample(self, batch_size):
#         if batch_size > len(self.memory):
#             batch_size = len(self.memory)
#         transitions = random.sample(self.memory, batch_size)
#         batch = Experience(*zip(*transitions))

#         # reset the memory
#         self.memory = []
#         self.position = 0
#         return batch

#     def __len__(self):
#         return len(self.memory)


# import random
# from collections import namedtuple

# # Experience 结构
# Experience = namedtuple("Experience",
#                         ("states", "actions", "returns", "advantages"))

# class OnPolicyReplayMemory(object):
#     """
#     Replay memory buffer for MAPPO (on-policy)
#     Stores: states, actions, returns (V_target), advantages (GAE)
#     """
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0

#     def _push_one(self, state, action, return_, advantage):
#         """Push a single step into memory"""
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Experience(state, action, return_, advantage)
#         self.position = (self.position + 1) % self.capacity

#     def push(self, states, actions, returns, advantages):
#         """
#         Push batch of transitions into memory
#         states: list of states
#         actions: list of actions
#         returns: list of returns (V_target)
#         advantages: list of advantages (GAE)
#         """
#         if isinstance(states, list):
#             for s, a, r, adv in zip(states, actions, returns, advantages):
#                 self._push_one(s, a, r, adv)
#         else:
#             # single step
#             self._push_one(states, actions, returns, advantages)

#     def sample(self, batch_size):
#         """Sample a batch and clear the memory (on-policy)"""
#         if batch_size > len(self.memory):
#             batch_size = len(self.memory)
#         transitions = random.sample(self.memory, batch_size)
#         batch = Experience(*zip(*transitions))

#         # clear memory after sampling
#         self.memory = []
#         self.position = 0
#         return batch

#     def __len__(self):
#         return len(self.memory)

# 扩展成同时存 reward 和 cost
Experience = namedtuple("Experience",
                        ("states", "actions",
                         "reward_returns", "reward_advantages",
                         "cost_returns", "cost_advantages"))
class OnPolicyReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _push_one(self, state, action, reward_return, reward_adv, cost_return, cost_adv):
        """Push a single step into memory"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action,
                                                reward_return, reward_adv,
                                                cost_return, cost_adv)
        self.position = (self.position + 1) % self.capacity

    def push(self, states, actions, reward_returns, reward_advantages,
             cost_returns, cost_advantages):
        """
        Push batch of transitions into memory
        states: list of states
        actions: list of actions
        reward_returns / reward_advantages: list
        cost_returns / cost_advantages: list
        """
        if isinstance(states, list):
            for s, a, r_ret, r_adv, c_ret, c_adv in zip(
                    states, actions, reward_returns, reward_advantages,
                    cost_returns, cost_advantages):
                self._push_one(s, a, r_ret, r_adv, c_ret, c_adv)
        else:
            self._push_one(states, actions,
                           reward_returns, reward_advantages,
                           cost_returns, cost_advantages)

    def sample(self, batch_size):
        """Sample a batch and clear the memory (on-policy)"""
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*transitions))

        # clear memory after sampling
        self.memory = []
        self.position = 0
        return batch

    def __len__(self):
        return len(self.memory)



class ReplayMemory(object):
    """
    Replay memory buffer
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _push_one(self, state, action, reward, next_state=None, done=None):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push(self, states, actions, rewards, next_states=None, dones=None):
        if isinstance(states, list):
            if next_states is not None and len(next_states) > 0:
                for s, a, r, n_s, d in zip(states, actions, rewards, next_states, dones):
                    self._push_one(s, a, r, n_s, d)
            else:
                for s, a, r in zip(states, actions, rewards):
                    self._push_one(s, a, r)
        else:
            self._push_one(states, actions, rewards, next_states, dones)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*transitions))

        return batch

    def __len__(self):
        return len(self.memory)