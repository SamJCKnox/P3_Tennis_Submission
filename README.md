# Multi Agent Tennis Submission
The aim of this task was to solve the tennis environment using multiple agent reinforcement learning methods.

## The Environment
The environment has a state space of 8, including positions and velocities of the ball and racket, and an action space of 2 continuous action, including moving forward and backward, and up and down. The environment is considered solved when the average score is greater than 0.5 for 100 episodes.


## Getting Started
The dependencies that are required can be installed using the following:

First, install conda: https://www.anaconda.com/distribution/#download-section

Next, create a new conda environment and activate

`conda create -n Tennis python=3.6.3 anaconda`

`conda activate Tennis`

Next install pytorch using:
`conda install pytorch=0.4.0 cuda80 -c pytorch`

And ml-agents ugin:
`pip install mlagents==0.4.0`

Finally, the environment and scripts are downloaded from

`git clone https://github.com/SamJCKnox/P3_Tennis_Submission.git`

## Instructions
The `P3_Tennis_Submission` notebook is the header which calls all scripts required to run. Run all sections to train the agent. Outputs will show how the agent is performing. The last section shows the agent evaluation.

The networks trained in the current outputs of the Jupyter Notebook are in `BenchmarkNetworks`, copy these into the root directory to view in the evaluation section.

`Report.md` shows the architecture of the networks with the hyperparamteres.

