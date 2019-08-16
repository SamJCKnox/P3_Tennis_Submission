import numpy as np
import random
import copy
from collections import namedtuple, deque
import importlib
import model

importlib.reload(model)
from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib import colors
from IPython.display import display, clear_output
import pdb

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4        # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 3e-10        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.state_size = args['state_size']
        self.action_size = args['action_size']
        self.seed = random.seed(args['random_seed'])
        self.batch_size = args['batch_size']
        self.batches_per_update = args['batches_per_update']
        self.buffer_size=args['buffer_size']
        self.lr_actor = args['lr_actor']
        self.lr_critic = args['lr_critic']
        self.weight_decay = args['weight_decay']

        self.gamma = args['gamma']
        self.tau = args['tau']



        # Actor Network (w/ Target Network)
        self.actor_local = Actor(args).to(device)
        self.actor_target = Actor(args).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(args).to(device)
        self.critic_target = Critic(args).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        # Noise process
        self.noise = OUNoise(self.action_size, self.seed)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)

        self.update_type = args['update_type']

        # Step counter for decideding when to learn
        self.counter = 0

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        #pdb.set_trace()
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            # Save experience / reward
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size and self.counter % self.batches_per_update == 0:
            #pdb.set_trace()
            self.learn()
        self.counter += 1

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions=[]
        #pdb.set_trace()
        for state in states:
            state = torch.from_numpy(state).float().to(device)
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


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.05, sigma=0.05): # Default: sigma=0.2, theta=0.15
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.action_dim = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
