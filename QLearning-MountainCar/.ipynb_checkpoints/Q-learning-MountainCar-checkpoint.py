import gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")
env.reset()



episodes = 20000
render_every = 100

learning_rate = 0.1
discount = 0.95 # how important we find future actions

epsilon = 0.95
decay_from = 1000

buckets_size = 20
discreet_os_size = [buckets_size] * len(env.observation_space.high)
discreet_os_win_size = (env.observation_space.high - env.observation_space.low) / discreet_os_size

#Create Q table
q_table = np.random.uniform(low=-2, high=0, size=(discreet_os_size + [env.action_space.n]))
def get_discreet_state(state):
    discreet_state = (state - env.observation_space.low) / discreet_os_win_size
    return tuple(discreet_state.astype(int))


prior_reward = 0
rewards = [0]

for episode in range(episodes):

    state = env.reset()
    discreet_state = get_discreet_state(state[0])
    done = False

    episode_reward = 0
    print(episode)
    
    while not done:
        
        #if np.random.random() > epsilon:
        if episode > 1000:
            action = np.argmax(q_table[discreet_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        
        new_state, reward, _, done, _ = env.step(action)
        new_discreet_state = get_discreet_state(new_state)
        current_q = q_table[discreet_state + (action,)]

        episode_reward += reward
        
        if not done:
            new_q = current_q + learning_rate * (reward + discount * np.max(q_table[new_discreet_state] - current_q))
            q_table[discreet_state] = new_q

    if episode % render_every == 0:
        env.render()

    rewards.append(episode_reward)

    if episode % render_every == 0:
        
        print(f"Episode: {episode} Reward: {rewards[episode]}")

    
    
env.close()


















