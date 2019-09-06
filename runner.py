import random
import torch
import numpy as np
from collections import deque

import importlib
import ddpg_agent
importlib.reload(ddpg_agent)
from ddpg_agent import Agent

def ddpg_runner(args):

    env = args['environment']
    brain_name = args['brain_name']
    scores = []
    agent = ddpg_agent.Agent(args['agent_args'])
    achievement = args['achievement']
    achievement_length = args['achievement_length']
    scores_deque = deque(maxlen=achievement_length)


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
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)),end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if np.mean(scores_deque)>achievement:
            return scores

    return scores
