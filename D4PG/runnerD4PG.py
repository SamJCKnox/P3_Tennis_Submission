#!/usr/bin/env python3
import os
import time
import ptan

import argparse
from tensorboardX import SummaryWriter
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque
import random


import pdb

GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000


TEST_ITERS = 1000



Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done', 'last_state'])



save_path = "C:\\Users\\mbgpcsk4\\Dropbox (The University of Manchester)\\D2019\\University\\Udacity\\P3_Tennis_Submission\\D4PG\\Checkpoint\\"

class D4PGRunner ():
    def __init__(self,args):
   
        self.env = args['environment']
        self.brain_name = args['brain_name']
        self.scores = []
        self.achievement = args['achievement']

        self.model_args = args['model_args']
        self.batch_size = args['batch_size']
        self.buffer_size=args['buffer_size']
        self.reward_steps = args['reward_steps']
        self.test_every = args['test_every']
        self.lr_actor = args['lr_actor']
        self.lr_critic = args['lr_critic']
        self.weight_decay = args['weight_decay']

        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']
        self.beta = args['beta']

        self.state_size = args['model_args']['state_size']
        self.action_size = args['model_args']['action_size']
        self.random_seed = args['model_args']['random_seed']
        self.fc1_units = args['model_args']['fc1_units']
        self.fc2_units = args['model_args']['fc2_units']
        self.N_atoms = args['model_args']['N_atoms']
        self.Vmin = args['model_args']['Vmin']
        self.Vmax = args['model_args']['Vmax']
        self.Delta_z = (self.Vmax - self.Vmin) / (self.N_atoms - 1)

        self.act_net = model.DDPGActor(self.model_args).to(device)
        self.crt_net = model.D4PGCritic(self.model_args).to(device)

        self.tgt_act_net = model.TargetNet(self.act_net)
        self.tgt_crt_net = model.TargetNet(self.crt_net)

        self.writer = SummaryWriter(comment="-d4pg_train")
        self.agent = agents.AgentDDPG(self.act_net, device=device)
        self.buffer = experience.PrioritizedReplayBuffer(buffer_size=self.buffer_size, alpha = self.alpha)
        self.act_opt = optim.Adam(self.act_net.parameters(), lr=self.lr_actor)
        self.crt_opt = optim.Adam(self.crt_net.parameters(), lr=self.lr_critic)


    def test_net(self,count=10, device="cpu"):
        rewards = 0.0
        steps = 0
        for _ in range(count):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations
            while True:
                actions = self.agent.__call__(states)
                env_info = self.env.step(actions)[self.brain_name]
                next_states = env_info.vector_observations
                reward = env_info.rewards
                dones = env_info.local_done
                states = next_states
                rewards += np.sum(reward)
                steps += 1
                if np.any(dones):
                    break
        return rewards / count, steps / count

    def run_net(self, count = 1):
        env_info = self.env.reset(train_mode=True)[self.brain_name]     
        states = env_info.vector_observations                  
        while True:
            actions = self.agent.__call__(states)
            env_info = self.env.step(actions)[self.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            for s,a,r,ns,d in zip(states, actions, rewards, next_states, dones):
                self.buffer._add(sample=Experience(s,a,r,d,ns))
            states = next_states
            if np.any(dones):
                break
        return


    def distr_projection(self,next_distr_v, rewards_v, dones_mask_t, gamma, device="cpu"):


        next_distr = next_distr_v.data.cpu().numpy()
        rewards = rewards_v.data.cpu().numpy()
        dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
        batch_size = len(rewards)
        proj_distr = np.zeros((batch_size, self.N_atoms), dtype=np.float32)

        for atom in range(self.N_atoms):
            tz_j = np.minimum(self.Vmax, np.maximum(self.Vmin, rewards + (self.Vmin + atom * self.Delta_z ) * self.gamma))
            b_j = (tz_j - self.Vmin) / self.Delta_z 
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
            ne_mask = u != l
            proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

        if dones_mask.any():
            proj_distr[dones_mask] = 0.0
            tz_j = np.minimum(self.Vmax, np.maximum(self.Vmin, rewards[dones_mask]))
            b_j = (tz_j - Vmin) / self.Delta_z 
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            eq_dones = dones_mask.copy()
            eq_dones[dones_mask] = eq_mask
            if eq_dones.any():
                proj_distr[eq_dones, l[eq_mask]] = 1.0
            ne_mask = u != l
            ne_dones = dones_mask.copy()
            ne_dones[dones_mask] = ne_mask
            if ne_dones.any():
                proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
                proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
        return torch.FloatTensor(proj_distr).to(device)


    def runnerD4PG(self):
        frame_idx = 0
        best_reward = None
        with ptan.common.utils.RewardTracker(self.writer) as tracker:
            with ptan.common.utils.TBMeanTracker(self.writer, batch_size=10) as tb_tracker:
                while True:
                    
                    self.run_net()

                    if len(self.buffer) < self.batch_size:
                        continue
                
                    batch = self.buffer.sample(self.batch_size, beta=self.beta)
                    states_v, actions_v, rewards_v, dones_mask, last_states_v = common.unpack_batch_ddqn(batch, device)

                    # train critic
                    self.crt_opt.zero_grad()
                    crt_distr_v = self.crt_net(states_v, actions_v)
                    last_act_v = self.tgt_act_net.target_model(last_states_v)
                    last_distr_v = F.softmax(self.tgt_crt_net.target_model(last_states_v, last_act_v), dim=1)
                    proj_distr_v = self.distr_projection(last_distr_v, rewards_v, dones_mask,
                                                    gamma=self.gamma**self.reward_steps, device=device)
                    prob_dist_v = -F.log_softmax(crt_distr_v, dim=1) * proj_distr_v
                    critic_loss_v = prob_dist_v.sum(dim=1).mean()
                    critic_loss_v.backward()
                    self.crt_opt.step()
                    tb_tracker.track("loss_critic", critic_loss_v, frame_idx)

                    # train actor
                    self.act_opt.zero_grad()
                    cur_actions_v = self.act_net(states_v)
                    crt_distr_v = self.crt_net(states_v, cur_actions_v)
                    actor_loss_v = -self.crt_net.distr_to_q(crt_distr_v)
                    actor_loss_v = actor_loss_v.mean()
                    actor_loss_v.backward()
                    self.act_opt.step()
                    tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                    self.tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                    self.tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                    if frame_idx %  self.test_every == 0:
                        ts = time.time()
                        rewards, steps = self.test_net(device=device)
                        print("Test: %d \t Time: %.2f \t Reward: %.3f \t Steps: %d" % (
                            frame_idx / self.test_every, time.time() - ts, rewards, steps))
                        self.writer.add_scalar("test_reward", rewards, frame_idx)
                        self.writer.add_scalar("test_steps", steps, frame_idx)
                        if best_reward is None or best_reward < rewards:
                            if best_reward is not None:
                                print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                                name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                                fname = os.path.join(save_path, name)
                                torch.save(self.act_net.state_dict(), fname)
                            best_reward = rewards


                    frame_idx +=1


        pass
