# Report: Tennis - MultiAgent

The method used to solve this is DDPG, a method analogous to an actor-critic method, where there is an actor neural network which maps the state directly to actions for each motor. This is performed using a double fully connected hidden layer network with relu activation functions and an output fully connected layer with a tanh activation function. This network during the update, is performing **gradient ascent** where the agent is trying to maximise the reward.
This method also has a critic which is a value based method, which judges the output of the actor and is used to calculate the updates for the actor. The network used in the critic is a fully connected layer, which then goes through a batch normalization and then is concatinated with the actor output, before going through another hidden layer and an output layer. All the activation functions in the critic are linear and relu functions.


## Modifications
The only code that actually worked was the DDPG code from project 2. I made modifications to a D4PG script and tried to implement prioritized experience replay (PER) however, this slowed the script by over 300%. The PER script was found in ptan, a useful pytorch library, however still ran too slowly and didn't find increase learning. These attempts are all found in `D4PG`.

### Hyperparameters
Network parameters apply to both actor and critic networks:
* fc1_units = 128
* fc2_units = 56

Agent parameters:
* tau = 1e-3
* gamma = 0.9
* buffer_size = int(2e6)
* batch_size = 1024
* batches_per_update = 10
* lr_actor = 1e-3
* lr_critic = 1e-3
* weight_decay = 0

## Final learning Scores
After all these implementations, the agent managed to achieve an average score across 30 consequtive episodes over all 20 arms after 132 episodes. The plot below shows the average of all 20 arms over the last 100 episodes, hence the average from 132 to 231 inclusive are greater than 30.

![alt text](#ToFillIn)

## Ideas for future work
Some ideas for future work include:
* Prioretized Experience Replay (PER) - Initially, the positive rewards are very sparce and should be learnt from more and hence it would be advantageous to learn from them more.
* N-step boot strapping - To aid with PER, being able to see actions and states from n-steps in the past would be likely speed up learning time as it would be possible to see the actions that lead to the reward.
* D4PG - Finally, implementing Distributed Distributional Deterministic Policy Gradient method would likely improve the rate of learning.


