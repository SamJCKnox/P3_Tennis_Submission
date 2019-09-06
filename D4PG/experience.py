import torch
import random
import collections


import numpy as np

from collections import namedtuple, deque

import utils



class ExperienceReplayBuffer:
    def __init__(self, args):
        
        self.buffer = []
        self.capacity = args['buffer_size']
        self.pos = 0
        self.device = args['device']
        self.batch_size = args['batch_size']

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self):
        """
        Get one random batch from experience replay
        TODO: implement sampling order policy
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= self.batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), self.batch_size, replace=True)
        samples = [self.buffer[key] for key in keys]
        state, action, reward, next_state, done = zip(*samples)

        # Stacks the experiences 
        states = torch.stack(state).to(self.device)
        actions = torch.stack(action).to(self.device)
        rewards = torch.stack(reward).to(self.device)
        next_states = torch.stack(next_state).to(self.device)
        dones = torch.stack(done).to(self.device)

        return states, actions, rewards, next_states, dones

    def add(self, sample):
        if len(self.buffer) < self.capacity:

            self.buffer.append(sample)

        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def ready(self):
        if len(self.buffer) < self.batch_size:
            return False
        else: 
            return True




class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, args):
        self.buffer_size = args['buffer_size']    
        self.batch_size = args['batch_size']     
        self._alpha = args['alpha']
        self.device = args['device']
        self.beta = args['beta']
        self.delta_beta = args['delta_beta']

        super(PrioritizedReplayBuffer, self).__init__(args)


        it_capacity = 1
        while it_capacity < self.buffer_size:
            it_capacity *= 2

        self._it_sum = utils.SumSegmentTree(it_capacity)
        self._it_min = utils.MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self.pos
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self):
        assert self.beta > 0

        idxes = self._sample_proportional(self.batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-self.beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-self.beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)
        samples = [self.buffer[idx] for idx in idxes]

        if self.beta < 1.0:
            self.beta *= self.delta_beta

        state, action, reward, next_state, done = zip(*samples)

        # Stacks the experiences 
        states = torch.stack(state).to(self.device)
        actions = torch.stack(action).to(self.device)
        rewards = torch.stack(reward).to(self.device)
        next_states = torch.stack(next_state).to(self.device)
        dones = torch.stack(done).to(self.device)

        return states, actions, rewards, next_states, dones


    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def ready(self):
        if len(self.buffer) < self.batch_size:
            return False
        else: 
            return True
