import numpy as np
import os
import gym
import matplotlib.pyplot as plt
import imageio

def create_model_asset_dir(model_folder_name="QLearning/CartPole-v1/models", asset_folder_name="QLearning/CartPole-v1/assets"):
    current_path = os.getcwd()
    model_dir = os.path.join(current_path, model_folder_name)
    asset_dir = os.path.join(current_path, asset_folder_name)
    print(model_dir)
    print(asset_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(asset_dir):
        os.makedirs(asset_dir)
    return model_dir, asset_dir

np.random.seed(42)

class QLearn:
    def __init__(self, env_name, buckets=50, qlow_init=-2, qhigh_init=0):
        self.qlow_init = qlow_init
        self.qhigh_init = qhigh_init
        self.env_name = env_name
        self.env = gym.make(env_name, render_mode=None)
        self.model_dir, self.asset_dir = create_model_asset_dir()
        self.episode_rewards = []
        self.discrete_os_size = [buckets] * len(self.env.observation_space.high)
        # self.discrete_os_size = list(map(lambda x: x + 15, self.discrete_os_size))
        # if env_name == "MountainCar-v0":
        self.discrete_os_win_size = (self.env.observation_space.high - self.env.observation_space.low) / self.discrete_os_size

        # self.discrete_os_win_size = [25,25,25,25,25,25]
        # elif env_name == "CartPole-v1":
        #     self.high = np.array([self.env.observation_space.high[0], 20, self.env.observation_space.high[2], 20])
        #     self.low = np.array([self.env.observation_space.low[0], -20, self.env.observation_space.low[2], -20])
        #     self.discrete_os_win_size = (self.high - self.low) / self.discrete_os_size

        

    def Learn(self, epsilon=0.95, decay_start=500, decay_end=15000, learning_rate=0.5, discount=0.995, episodes=20_000, logs_every=100, mean_rew=100, win=195):
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
                self.new_state, self.reward, self.truncation, self.termination, _ = self.env.step(self.action)
                self.done = self.truncation or self.termination
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

    def load_bestqtable(self):
        self.q_table = np.load(self.bestq_dir)

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

    def save_gif(self, num_episodes=5):
        env = gym.make(self.env_name, render_mode="rgb_array")
        images = []
        for _ in range(num_episodes):
            state = env.reset()
            discrete_state = self.get_discrete_state(state[0])
            done = False
            while not done:
                images.append(env.render())  # Capture the frame
                action = np.argmax(self.best_model[discrete_state])
                new_state, _, truncation, termination, _ = env.step(action)
                done = truncation or termination
                new_discrete_state = self.get_discrete_state(new_state)
                discrete_state = new_discrete_state
        gif_path = os.path.join(self.asset_dir, "solved_gameplay.gif")
        imageio.mimsave(gif_path, images, fps=30)



    def save_best_model(self):
        self.bestq_dir = os.path.join(self.model_dir, "best_qtable" + '.npy')
        np.save(self.bestq_dir, self.best_model)
        print(f"Models saved. Mean score: {self.win}")


    def plot_save_rewards_graph(self):
        moving_average = np.convolve(self.episode_rewards, np.ones(self.mean_c)/self.mean_c, mode='valid')
        plt.figure(figsize=(10, 5))
        plt.plot(moving_average, label='Average Reward per 100 Episodes')
        plt.title('Moving Average of Rewards Over 100 Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(self.asset_dir, "mean_rewards_graph.png")
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
        if not self.termination:
            max_future_q = np.max(self.q_table[new_discrete_state + (action,)])
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (self.reward + self.discount_rate * max_future_q)
            self.q_table[discrete_state + (action,)] = new_q
        elif self.truncation:
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