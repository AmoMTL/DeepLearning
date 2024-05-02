# Deep Deterministic Policy Gradient

Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy. [[source]](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

<div style="background-color: white; display: inline-block;">
    <img align="center" src="https://spinningup.openai.com/en/latest/_images/math/3a8b6ce0d6c0b68744b5724403f5d70ed5cda5db.svg">
</div>



### Points:
* DDPG is an off-policy algorithm.
* DDPG can only be used for environments with continuous action spaces.
* DDPG can be thought of as being deep Q-learning for continuous action spaces.
* The Spinning Up implementation of DDPG does not support parallelization.