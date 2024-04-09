import gym
import numpy as np
import math



env = gym.make("MountainCar-v0", render_mode=None)


episodes = 20000
render_every = 1000

logs_every = 100

learning_rate = 0.1
discount = 0.95 # how important we find future actions

epsilon = 0.95
decay_start = 1000
decay_end = 8000
decay_value = epsilon / (decay_end - decay_start)

buckets_size = 20
discreet_os_size = [buckets_size] * len(env.observation_space.high)
discreet_os_win_size = (env.observation_space.high - env.observation_space.low) / discreet_os_size

#Create Q table
q_table = np.random.uniform(low=-2, high=0, size=(discreet_os_size + [env.action_space.n]))
def get_discreet_state(state):
    discreet_state = (state - env.observation_space.low) / discreet_os_win_size
    return tuple(discreet_state.astype(int))


prior_reward = -200
rewards = [prior_reward]

for episode in range(episodes):

    
    state = env.reset()
    discreet_state = get_discreet_state(state[0])
    done = False

    new_episode_reward = 0
    
    if episode % logs_every == 0:
        mean_reward = sum(rewards[-logs_every:]) / logs_every
        print(f"Episode: {episode} Mean Reward: {mean_reward} Epsilon: {epsilon}")
    
    while not done:
        
        if np.random.random() > epsilon:
        #if episode > 1000:
            action = np.argmax(q_table[discreet_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        
        new_state, reward, truncation, termination, _ = env.step(action)
        done = truncation or termination
        new_discreet_state = get_discreet_state(new_state)
        current_q = q_table[discreet_state + (action,)]

        new_episode_reward += reward
        
        if not done:
            max_future_q = np.max(q_table[new_discreet_state + (action,)])
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[discreet_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discreet_state, (action, )] = 0
        
        discreet_state = new_discreet_state
            #print(discreet_state)

    if decay_start <= episode <= decay_end:
        epsilon -= decay_value

    prior_reward = new_episode_reward
    
    if episode % render_every == 0:
        
        print(f"Episode: {episode} Reward: {rewards[episode]} Epsilon={epsilon}")
        
        render_done = False
        render_env = gym.make("MountainCar-v0", render_mode="human")
        rstate = render_env.reset()
        rdiscreet_state = get_discreet_state(rstate[0])
        while not render_done:
            action = np.argmax(q_table[rdiscreet_state])
            rnew_state, rreward, rtruncation, rtermination, _ = render_env.step(action)
            render_done = rtermination or rtruncation
            rnew_discreet_state = get_discreet_state(rnew_state)
            rdiscreet_state = rnew_discreet_state

        print(f"Render Episode Reward: {rreward} Epsilon={epsilon}")
        
        render_env.close()

    rewards.append(new_episode_reward)


env.close()


















