import numpy as np
import os
import gym
import matplotlib.pyplot as plt
import imageio

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

np.random.seed(42)

class QLearn:
    def __init__(self, env, buckets=50, qlow_init=-2, qhigh_init=0):
        self.qlow_init = qlow_init
        self.qhigh_init = qhigh_init
        self.env = env
        self.episode_rewards = []
        self.discrete_os_size = [buckets] * len(self.env.observation_space.high)
        self.discrete_os_win_size = (self.env.observation_space.high - self.env.observation_space.low) / self.discrete_os_size
        

    def Learn(self, epsilon=0.95, decay_start=500, decay_end=15000, learning_rate=0.5, discount=0.995, episodes=20_000, logs_every=100, mean_rew=100, win=-110):
        self.learning_rate = learning_rate
        self.win = win
        self.discount_rate = discount
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.logs_every = logs_every
        self.mean_c = mean_rew
        self.epsilon = epsilon
        self.decay_value = self.epsilon / (self.decay_end - self.decay_start)
        self.q_table = self.initialize_Qtable()
        
        
        for episode in range(episodes):
            self.state = self.env.reset()
            self.discrete_state = self.get_discrete_state(self.state[0])
            self.done = False
        
            episode_reward = 0
            while not self.done:
                self.action = self.get_action()
                self.new_state, self.reward, truncation, termination, _ = self.env.step(self.action)
                self.done = truncation or termination
                episode_reward += self.reward
                self.new_discrete_state = self.get_discrete_state(self.new_state)
                self.update_Qtable(self.discrete_state, self.new_discrete_state, self.action)
                self.discrete_state = self.new_discrete_state
            
            self.decay_epsilon(episode)
            self.episode_rewards.append(episode_reward)

            if episode >= logs_every and episode % logs_every == 0:
                self.display_log()

            if episode > 1000:
                mean_reward = sum(self.episode_rewards[-self.mean_c:]) / self.mean_c
                if mean_reward >= self.win:
                    self.best_model = self.q_table
                    self.win = mean_reward

    def load_qtable(self, qtable_dir="QLearning-MountainCar/models/best_qtable.npy"):
        self.q_table = np.load(qtable_dir)

    def render(self, num_episodes=5):
        for episode in range(num_episodes):
            state = self.env.reset()
            discrete_state = self.get_discrete_state(state[0])
            done = False
            while not done:
                action = np.argmax(self.q_table[discrete_state])
                new_state, _, truncation, termination, _ = self.env.step(action)
                done = truncation or termination
                new_discrete_state = self.get_discrete_state(new_state)
                discrete_state = new_discrete_state

    def save_gif(self, env_name="MountainCar-v0", num_episodes=5, gif_path="QLearning-MountainCar/assets/gameplay.gif"):
        env = gym.make(env_name, render_mode="rgb_array")
        images = []
        for _ in range(num_episodes):
            state = env.reset()
            discrete_state = self.get_discrete_state(state[0])
            done = False
            while not done:
                images.append(env.render())  # Capture the frame
                action = np.argmax(self.q_table[discrete_state])
                new_state, _, truncation, termination, _ = env.step(action)
                done = truncation or termination
                new_discrete_state = self.get_discrete_state(new_state)
                discrete_state = new_discrete_state
        imageio.mimsave(gif_path, images, fps=30)



    def save_best_model(self, save_dir="QLearning-MountainCar/models"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "best_qtable" + '.npy')
        np.save(save_path, self.best_model)
        print(f"Models saved. Mean score: {self.win}")


    def plot_save_rewards_graph(self, save_dir="QLearning-MountainCar/graphs"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        moving_average = np.convolve(self.episode_rewards, np.ones(self.mean_c)/self.mean_c, mode='valid')
        plt.figure(figsize=(10, 5))
        plt.plot(moving_average, label='Average Reward per 100 Episodes')
        plt.title('Moving Average of Rewards Over 100 Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(save_dir, "mean_rewards_graph.png")
        plt.savefig(save_path)
        plt.close()


    def display_log(self):
        print(f"Episodes:       {len(self.episode_rewards)}\n")
        print(f"Mean rewards:   {sum(self.episode_rewards[-self.mean_c:]) / self.mean_c}\n")
                
    def decay_epsilon(self, episode):
        if self.decay_start <= episode <= self.decay_end:
            self.epsilon -= self.decay_value

    def update_Qtable(self, discrete_state, new_discrete_state, action):
        current_q = self.q_table[discrete_state + (action,)]
        if not self.done:
            max_future_q = np.max(self.q_table[new_discrete_state + (action,)])
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (self.reward + self.discount_rate * max_future_q)
            self.q_table[discrete_state + (action,)] = new_q
        elif self.new_state[0] >= self.env.goal_position:
            self.q_table[discrete_state, (action, )] = 0

    def get_action(self):
        if np.random.random() > self.epsilon:
            action = np.argmax(self.q_table[self.discrete_state])
            return action
        else:
            action = np.random.randint(0, self.env.action_space.n)
            return action

    def get_discrete_state(self, state):
        discrete_state = (state - self.env.observation_space.low) / self.discrete_os_win_size
        return tuple(discrete_state.astype(int))

    def initialize_Qtable(self):
        q_table = np.random.uniform(low=self.qlow_init, high=self.qhigh_init, size=(self.discrete_os_size + [self.env.action_space.n]))
        return q_table