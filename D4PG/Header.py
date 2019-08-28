import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib
import runnerD4PG

from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name="C:\\Users\\mbgpcsk4\\Dropbox (The University of Manchester)\\D2019\\University\\Udacity\\P3_Tennis_Submission\\D4PG\\Tennis_Windows_x86_64\\Tennis.exe", no_graphics = True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

arguments = {
    'episodes': 10000,                           # number of episodes
    'brain_name': brain_name,                   # the brain name of the unity environment
    'achievement': 10.,                         # score at which the environment is considered beaten
    'achievement_length': 300,                  # how long the agent needs to get a score above the achievement to solve the environment
    'environment': env, 
    'update_type': 'soft',
    'tau': 1e-3,
    'gamma': 0.9,
    'buffer_size': int(3e6),
    'batch_size': 1024,
    'batches_per_update': 1,
    'lr_actor': 1e-3,
    'lr_critic': 1e-3,
    'weight_decay': 0,
    'alpha': 0.9,
    'beta':0.9,
    'model_args': {  
        'state_size': 24,
        'action_size': 2,
        'random_seed': 9,
        'fc1_units': 128,
        'fc2_units': 56,
        'N_atoms':51, 
        'Vmin':-10, 
        'Vmax':10,
        
        },
    }

import runnerD4PG
importlib.reload(runnerD4PG)

scores = runnerD4PG.runnerD4PG(arguments)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()