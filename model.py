import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = (1. / np.sqrt(fan_in))**2
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, args):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(args['random_seed'])
        self.fc1 = nn.Linear(args['state_size'], args['fc1_units'])
        self.fc2 = nn.Linear(args['fc1_units'], args['fc2_units'])
        self.fc3 = nn.Linear(args['fc2_units'], args['action_size'])
        #self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, args):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(args['random_seed'])
        self.fcs1 = nn.Linear(args['state_size'], args['fc1_units'])
        self.bn1 = nn.BatchNorm1d(num_features=args['fc1_units'])
        self.fc2 = nn.Linear(args['fc1_units']+args['action_size'], args['fc2_units'])
        self.fc3 = nn.Linear(args['fc2_units'], 1)
        #self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
