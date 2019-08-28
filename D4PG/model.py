import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPGActor(nn.Module):
    def __init__(self, args):
        super(DDPGActor, self).__init__()
        
        self.state_size=args['state_size']
        self.action_size = args['action_size']
        self.fc1_units = args['fc1_units']
        self.fc2_units = args['fc2_units']


        self.net = nn.Sequential(
            nn.Linear(self.state_size, self.fc1_units),
            nn.ReLU(),
            nn.Linear(self.fc1_units, self.fc2_units),
            nn.ReLU(),
            nn.Linear(self.fc2_units, self.action_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class D4PGCritic(nn.Module):
    def __init__(self, args):
        super(D4PGCritic, self).__init__()

        self.state_size=args['state_size']
        self.action_size = args['action_size']
        self.fc1_units = args['fc1_units']
        self.fc2_units = args['fc2_units']
        self.N_atoms=args['N_atoms']
        self.Vmin = args['Vmin']
        self.Vmax = args['Vmax']


        self.obs_net = nn.Sequential(
            nn.Linear(self.state_size, self.fc1_units),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(self.fc1_units + self.action_size, self.fc2_units),
            nn.ReLU(),
            nn.Linear(self.fc2_units, self.N_atoms)
        )

        self.delta = (self.Vmax - self.Vmin) / (self.N_atoms - 1)
        self.register_buffer("supports", torch.arange(self.Vmin, self.Vmax + self.delta, self.delta))

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)



class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


        