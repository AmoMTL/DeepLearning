# MountainCar-v0
<img align="right" width="200" src="https://user-images.githubusercontent.com/8510097/31701297-3ebf291c-b384-11e7-8289-24f1d392fb48.PNG">A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.
* [Environment details](https://github.com/openai/gym/wiki/MountainCar-v0)
* *MountainCar-v0 defines "solving" as getting average reward of -110.0 over 100 consecutive trials.*

## Solved using QLearning algorithm
<img align="right" width="400" src="assets/mean_rewards_graph.png">I implemented the Q learning algorithm to scores for the best action at different states of the game. The algorithm achieves "solve" in ~6000 episodes, as can be seen in the graph to the left.

The [best model](models) acheived a score of *-97.61* over 100 consecutive episodes.




<br>
<br>

### To train
```console
py -m train.py
```
The best model from the training is saved in the `models` folder and an image of the plot between the mean reward over 100 episodes and the total number of episodes is saved in the `graphs` folder.