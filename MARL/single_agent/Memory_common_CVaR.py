import random
from collections import namedtuple

# 扩展成同时存 reward 和 cost
Experience = namedtuple("Experience",("states", "actions","reward_returns", "cost"))
class OnPolicyReplayMemoryCVaR(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _push_one(self, state, action, reward_return, cost):
        """Push a single step into memory"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action,reward_return,cost)
        self.position = (self.position + 1) % self.capacity

    def push(self, states, actions, reward_returns,cost):
        """
        Push batch of transitions into memory
        states: list of states
        actions: list of actions
        reward_returns / reward_advantages: list
        cost_returns / cost_advantages: list
        """
        if isinstance(states, list):
            for s, a, r_ret,  c in zip(states, actions, reward_returns,cost):
                self._push_one(s, a, r_ret,  c)
        else:
            self._push_one(states, actions, reward_returns,cost)

    def sample(self, batch_size):
        """Sample a batch and clear the memory (on-policy)"""
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = self.memory[:batch_size]
        batch = Experience(*zip(*transitions))

        # clear memory after sampling
        self.memory = []
        self.position = 0
        return batch

    def __len__(self):
        return len(self.memory)



