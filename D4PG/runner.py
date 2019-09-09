## D4PG Imports

#!/usr/bin/env python3
import os
import ptan
import time
import argparse

import numpy as np

import torch
import torch.optim as optim



## Original imports


import random
import torch
import numpy as np
from collections import deque


def ddpg_runner(args):
    env = args['environment']
    brain_name = args['brain_name']

    agent = args['agent']
    memory = args['memory']
    device = args['device']

    achievement = args['achievement']
    scores_deque = deque(maxlen=args['achievement_length'])
    scores = []

    for i_episode in range(1, args['episodes']+1):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
        states = env_info.vector_observations                  # get the current state (for each agent)
        score = 0
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done                        # see if episode finished
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += np.mean(rewards)
            if np.any(dones):
                break

        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)),end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)))

        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if np.mean(scores_deque)>achievement:
            return scores

    return scores
