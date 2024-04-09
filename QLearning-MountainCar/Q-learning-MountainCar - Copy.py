import gym
import numpy as np



env = gym.make("MountainCar-v0", render_mode=None)


episodes = 20000
render_every = 1000

learning_rate = 0.1
discount = 0.95 # how important we find future actions

epsilon = 0.95
decay_start = 100
decay_end = 8000
epsilon_decay = epsilon / (decay_end - decay_start)

buckets_size = 40
discreet_os_size = [buckets_size] * len(env.observation_space.high)
discreet_os_win_size = (env.observation_space.high - env.observation_space.low) / discreet_os_size

#Create Q table
#q_table = np.random.uniform(low=-2, high=0, size=(discreet_os_size + [env.action_space.n]))
q_table = np.zeros(shape=(discreet_os_size + [env.action_space.n]))

def get_discreet_state(state):
    discreet_state = (state - env.observation_space.low) / discreet_os_win_size
    return tuple(discreet_state.astype(int))

obj_end_discreet_state = ((env.goal_position - env.observation_space.low[0]) / discreet_os_win_size[0]).astype(int)


prior_reward = 0
rewards = [0]

for episode in range(episodes):

    
    state = env.reset()
    discreet_state = get_discreet_state(state[0])
    done = False

    new_episode_reward = 0
    #print(f"Episode: {episode} Reward: {rewards[episode]}")
    
    while not done:
        
        if np.random.random() > epsilon:
        #if episode > 1000:
            action = np.argmax(q_table[discreet_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        
        new_state, _, truncation, termination, _ = env.step(action)
        done = truncation or termination

        new_discreet_state = get_discreet_state(new_state)

        # Create reward function
        delta = obj_end_discreet_state - new_discreet_state[0]
        reward = 1 / delta

        current_q = q_table[discreet_state + (action,)]

        new_episode_reward += reward
        
        if not done:
            max_future_q = np.max(q_table[discreet_state + (action,)])
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[discreet_state + (action,)] = new_q
        elif new_state[0] >=env.goal_position:
            q_table[discreet_state, (action, )] = 0
        
        discreet_state = new_discreet_state
            #print(discreet_state)

    if decay_start >= episode >= decay_end:
        epsilon -= epsilon_decay

    prior_reward = new_episode_reward
    
    if episode % render_every == 0:
        
        print(f"Episode: {episode} Reward: {rewards[episode]} Epsilon={epsilon}")
        
        render_done = False
        render_env = gym.make("MountainCar-v0", render_mode="human")
        rstate = render_env.reset()
        rdiscreet_state = get_discreet_state(rstate[0])
        while not render_done:
            action = np.argmax(q_table[rdiscreet_state])
            rnew_state, _, rtruncation, rtermination, _ = render_env.step(action)
            render_done = rtermination or rtruncation
            rnew_discreet_state = get_discreet_state(rnew_state)
            rdiscreet_state = rnew_discreet_state
        
        render_env.close()

    rewards.append(new_episode_reward)


env.close()


















