# Report: Tennis - MultiAgent

The method used to solve this is DDPG, a method analogous to an actor-critic method, where there is an actor neural network which maps the state directly to actions for each motor. This is performed using a double fully connected hidden layer network with relu activation functions and an output fully connected layer with a tanh activation function. This network during the update, is performing **gradient ascent** where the agent is trying to maximise the reward.
This method also has a critic which is a value based method, which judges the output of the actor and is used to calculate the updates for the actor. The network used in the critic is a fully connected layer, which then goes through a batch normalization and then is concatenated with the actor output, before going through another hidden layer and an output layer. All the activation functions in the critic are linear and relu functions.


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
After all these implementations, the agent managed to achieve an average score of 1 across 100 consecutive episodes from episode 1433 to 1532 inclusive. The plot below shows the scores of each episode. There is high variance in the results, and if the code is left to run, the moving average over 100 episodes fluctuates between 0.3 and 1.3 which means that the algorithm isn't optimal, however still solves the environment to the level required.

![alt text](https://github.com/SamJCKnox/P3_Tennis_Submission/blob/master/Scores.JPG)

## Ideas for future work
Previously, I stated that the future work from the second project would include:
* Prioritized Experience Replay (PER) - Initially, the positive rewards are very sparse and should be learnt from more and hence it would be advantageous to learn from them more.
* N-step boot strapping - To aid with PER, being able to see actions and states from n-steps in the past would be likely speed up learning time as it would be possible to see the actions that lead to the reward.
* D4PG - Finally, implementing Distributed Distributional Deterministic Policy Gradient method would likely improve the rate of learning.

This submission is the second attempt at PER and was again, unsuccessful. Using cProfile, it was found that there was a great deal of time being used to calculate the priorities for each experience, even when the buffer was reduced to a few thousand experiences. An attempt at D4PG was also attempted, but was unable to complete the implementation as I didn't understand the complex implementations found online. Therefore, if I was to re do this assignment, I would look into different types of algorithms. One that is of interest is as follows:
* Perform a step in the environment
* Compare this experience with others in the replay buffer
* If the experience is adequately novel, include it in the buffer and give it a fixed priority proportional to the difference to the rest of the buffer and inversely proportional to the number of times it is revisited.

Later, when sampling the buffer, the samples are chosen based on priority. The main difference is that the priorities are not recalculated every sample. Hopefully, this would speed up the process.


