import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPGActor(nn.Module):
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

        super(DDPGActor, self).__init__()
        
        self.state_size=args['state_size']
        self.action_size = args['action_size']
        self.fc1_units = args['fc1_units']
        self.fc2_units = args['fc2_units']


        self.network = nn.Sequential(
            nn.Linear(self.state_size, self.fc1_units),
            nn.ReLU(),
            nn.Linear(self.fc1_units, self.fc2_units),
            nn.ReLU(),
            nn.Linear(self.fc2_units, self.action_size),
            nn.Tanh()
        )

        

    def forward(self, x):
        return self.network(x)


class DDPGCritic(nn.Module):
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
        super(DDPGCritic, self).__init__()

                
        self.state_size=args['state_size']
        self.action_size = args['action_size']
        self.fc1_units = args['fc1_units']
        self.fc2_units = args['fc2_units']


        self.first_net = nn.Sequential(
            nn.Linear(self.state_size, self.fc1_units),
            nn.ReLU(),
        )

        self.last_net = nn.Sequential(
            nn.Linear(self.fc1_units + self.action_size , self.fc2_units),
            nn.ReLU(),
            nn.Linear(self.fc2_units, 1)
        )


    def forward(self, x, a): 
        """Run forwards through the nerual network
        Params
        ======
            x (int): forward state
            a (int): actions chosen by actor

        """
        x = self.first_net(x)
        return self.last_net(torch.cat([x, a], dim=1))



class TargetNet(nn.Module):
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        super(TargetNet, self).__init__()
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


        