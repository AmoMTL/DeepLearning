# CartPole-v1

<img align="right" width="300" src="https://www.gymlibrary.dev/_images/cart_pole.gif">A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

* [Environment Details](https://github.com/openai/gym/wiki/CartPole-v0)

* *CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.*

## Solved using the QLearning algorithm

<img align="right" width="400" src="assets/mean_rewards_graph.png">I implemented the Q learning algorithm to scores for the best action at different states of the game. The algorithm achieves "solve" as given in the Open AI Gym [*wiki*](https://github.com/openai/gym/wiki/CartPole-v0) in ~25000 episodes, as can be seen in the graph to the left.

The [best model](models) acheived a score of *396.11* over 100 consecutive episodes.

### Gameplay by using actions from the trained Q table:
We can see the pole appears almost stationary and upright continuously.
<div align="center">
    <img width="400" src="assets/solved_gameplay.gif">
</div>

