import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import colors
from IPython.display import display, clear_output
import pdb


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, args):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
      
        
        self.batches_per_update = args['batches_per_update']
        self.lr_actor = args['lr_actor']
        self.lr_critic = args['lr_critic']
        self.weight_decay = args['weight_decay']
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.device=args['device']
        self.batch_size=args['batch_size']
        self.beta = args['beta']
   


        # Actor Network (w/ Target Network)
        self.actor_local = args['act_net']
        self.actor_target = args['tgt_act_net']
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = args['crt_net']
        self.critic_target = args['tgt_crt_net']
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        # Noise process
        self.noise = args['noise']

        # Replay memory
        self.memory = args['memory']

        self.update_type = args['update_type']

        # Step counter for decideding when to learn
        self.counter = 0

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        #pdb.set_trace()
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            # Save experience / reward
            state = torch.tensor(state).float()
            action = torch.tensor(action).float()
            reward = torch.tensor(reward).float()
            next_state = torch.tensor(next_state).float()
            done = torch.tensor(done).float()
            self.memory.add((state, action, reward, next_state, done))

        # Learn, if enough samples are available in memory
        if self.memory.ready() and self.counter % self.batches_per_update == 0:
            self.learn()
        self.counter += 1

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions=[]
        #pdb.set_trace()
        for state in states:
            state = torch.from_numpy(state).float().to(self.device)
            self.actor_local.eval()
            with torch.no_grad():
                action = self.actor_local(state).cpu().data.numpy()
            self.actor_local.train()
            if add_noise:
                action += self.noise.sample()
            actions.append(np.clip(action, -1, 1))
        return actions

    def reset(self):
        self.noise.reset()
        self.actor_local.reset_parameters
        self.actor_target.reset_parameters

    def print_params(self):
        print("Actor:\n")
        for p in self.actor_target.parameters():
            if p.requires_grad:
                 print(p.name, p.data)

        print("\nCritic:\n")
        for p in self.critic_target.parameters():
            if p.requires_grad:
                 print(p.name, p.data)


    def learn(self):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        #pdb.set_trace()
        for i in range(self.batches_per_update):

            
            states, actions, rewards, next_states, dones = self.memory.sample()

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = self.critic_local(states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        if self.update_type == 'soft':
            self.soft_update(self.critic_local, self.critic_target, self.tau)
            self.soft_update(self.actor_local, self.actor_target, self.tau)
        else:
            self.hard_update(self.critic_local, self.critic_target)
            self.hard_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())
        target_model.load_state_dict(local_model.state_dict())

