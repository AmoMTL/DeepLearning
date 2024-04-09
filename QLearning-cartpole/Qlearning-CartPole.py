import gym
import numpy as np
import random

env = gym.make("CartPole-v1",render_mode=None)
state = env.reset()

episodes = 100_000
render_every = 5000
logs_every = 1000
mean_period = 100

learning_rate = 0.1
discount = 0.95

epsilon = 0.95
decay_start = 5000
decay_end = 25000
decay_value = epsilon / (decay_end - decay_start)
epsilon_min = 0.01

bucket_size = 100
discrete_obs_size = [bucket_size] * len(env.observation_space.high)
high = np.array([env.observation_space.high[0], 10, env.observation_space.high[2], 10])
low = np.array([env.observation_space.low[0], -10, env.observation_space.low[2], -10])
discrete_obs_win_size = (high - low) / discrete_obs_size

q_table = np.zeros(shape=(discrete_obs_size + [env.action_space.n]))
print(q_table.shape)
def get_discrete_state(state):
    discrete_state = (state - low) / discrete_obs_win_size
    return tuple(discrete_state.astype(int))

rewards = []

for episode in range(episodes):

    state = env.reset()
    discrete_state = get_discrete_state(state[0])
    done = False
    episode_reward = 0

    if epsilon < epsilon_min:
        epsilon = epsilon_min

    if episode % logs_every == 0 and episode != 0:
        #print(episode)
        #print(len(rewards))    
        mean_reward = sum(rewards[-mean_period:]) / mean_period
        print(f"Episode: {episode} Mean reward: {mean_reward} Epsilon: {epsilon}")

    while not done:

        if np.random.random() < epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = random.randint(0, env.action_space.n - 1)
        
        new_state, reward, truncation, termination, _ = env.step(action)
        done = termination or truncation
        new_discrete_state = get_discrete_state(new_state)
        reward += 1
        episode_reward += reward
        current_q = q_table[discrete_state + (action,)]

        if not done:
            max_future_q = np.max(q_table[new_discrete_state + (action,)])
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state

    rewards.append(episode_reward)

    if decay_start <= episode <= decay_end:
        epsilon -= decay_value

    if episode % render_every == 0:
        render_env = gym.make("CartPole-v1",render_mode="human")
        rstate = render_env.reset()
        rdiscrete_state = get_discrete_state(rstate[0])
        rdone = False
        repisode_reward = 0
        while not rdone:
            raction = np.argmax(q_table[rdiscrete_state])
            rnew_state, rreward, rtermination, rtruncation, _ = render_env.step(raction)
            rdiscrete_state = get_discrete_state(rnew_state)
            rdone = rtermination or rtruncation
            repisode_reward += rreward
        print(f"Render Episode Reward: {repisode_reward}")
        render_env.close()


env.close()