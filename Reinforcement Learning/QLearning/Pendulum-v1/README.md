# Pendulum-v1
<img align="right" width="200" src="https://www.gymlibrary.dev/_images/pendulum.gif">A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.
* [Environment details](https://www.gymlibrary.dev/environments/classic_control/pendulum/)
* *According to the Open AI Gym [wiki](https://github.com/openai/gym/wiki/Leaderboard) Pendulum-v0 is an unsolved environment, which means it does not have a specified reward threshold at which it's considered solved, so I will consider obtaining a score of -150 to be "solved".*

## Solved using QLearning algorithm
<img align="right" width="400" src="assets/mean_rewards_graph.png">I implemented the Q learning algorithm to scores for the best action at different states of the game. The algorithm achieves "solve" in ~6000 episodes, as can be seen in the graph to the left.

The [best model](models) acheived a score of *-* over 100 consecutive episodes.




<br>
<br>

### To train
```console
py -m train.py
```
The best model from the training is saved in the `models` folder and an image of the plot between the mean reward over 100 episodes and the total number of episodes is saved in the `graphs` folder.