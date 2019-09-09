import numpy as np
import random
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, args):
        """Initialize parameters and noise process."""

        self.mu = args['mu'] * np.ones(args['action_size'])
        self.theta = args['theta']
        self.sigma = args['sigma']
        self.seed = random.seed(args['seed'])
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
