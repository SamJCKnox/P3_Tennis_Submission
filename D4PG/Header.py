import torch
import numpy as np
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment

from torchsummary import summary

# Python Code imports

from experience import PrioritizedReplayBuffer, ExperienceReplayBuffer
from runner import ddpg_runner
from model import DDPGActor, DDPGCritic, TargetNet
from noise import OUNoise
from agent import Agent

# Profiling
import cProfile
import re


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = "C:\\Users\\mbgpcsk4\\Dropbox (The University of Manchester)\\D2019\\University\\Udacity\\P3_Tennis_Submission\\D4PG\\Tennis_Windows_x86_64\\Tennis.exe"
env = UnityEnvironment(file_name=path , no_graphics = True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

# Create a memory buffer

arguments_memory = {
    'buffer_size': int(3e7),
    'batch_size': 1024,
    'alpha': 0.5,
    'beta': 0.5,
    'delta_beta': 1.0001,
    'device': device
    }
#memory = PrioritizedReplayBuffer(arguments_memory) ###########PER#############
memory = ExperienceReplayBuffer(arguments_memory)
arguments_model = {
    'state_size': 8,
    'action_size': 2,
    'random_seed': 865387,
    'fc1_units': 128,
    'fc2_units': 56,
    }

act_net = DDPGActor(arguments_model).to(device)
crt_net = DDPGCritic(arguments_model).to(device)

tgt_act_net = DDPGActor(arguments_model).to(device)
tgt_crt_net = DDPGCritic(arguments_model).to(device)

#tgt_act_net = TargetNet(act_net)
#tgt_crt_net = TargetNet(act_net)

## Print summary of network

summary(act_net, input_size=(arguments_model['state_size'],))


arguments_noise = {
    'mu': 0,
    'theta':  0.05,
    'sigma': 0.15,
    'seed': 5960,
    'action_size': 2
    }

noise = OUNoise(arguments_noise)

arguments_agent = {
    'batches_per_update':10,
    'lr_actor': 1e-3,
    'lr_critic': 1e-3,
    'gamma': 0.9,
    'weight_decay': 0,
    'tau': 1e-3,
    'act_net': act_net,
    'crt_net': crt_net,
    'tgt_act_net': tgt_act_net,
    'tgt_crt_net': tgt_crt_net,
    'memory': memory,
    'device': device,
    'noise': noise,
    'update_type': 'soft',
    'beta': 0.9,
    'batch_size': arguments_memory['batch_size']
    }

agent = Agent(arguments_agent)

arguments = {
    'episodes': 10000,                           # number of episodes
    'brain_name': brain_name,                   # the brain name of the unity environment
    'achievement': 2.,                         # score at which the environment is considered beaten
    'achievement_length': 100,                  # how long the agent needs to get a score above the achievement to solve the environment
    'environment': env, 
    'agent':agent,
    'memory': memory,
    'device': device,
    
    }


pr = cProfile.Profile()
pr.enable()
scores = ddpg_runner(arguments)
pr.disable()
pr.print_stats()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()