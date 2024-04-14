import gym
import numpy as np
import random
import os
import time
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from collections import deque

EPISODES = 20_000

BUCKETS_SIZE = 20
REPLAY_MEMORY_SIZE = 50_000 # how many steps to keep in the replay memory
MIN_REPLAY_MEMORY_SIZE = 1000 # minimum number of steps in a replay memory to start training
MINI_BATCH_SIZE = 64

EPSILON = 0.50
decay_start = 10
decay_end = 125
decay = EPSILON/(decay_end - decay_start)

MODEL_DIR = "models"
if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

LOGS_DIR = "logs"
if not os.path.isdir(LOGS_DIR):
    os.makedirs(LOGS_DIR)

def SaveArray(array, filename):
    with open(f"{LOGS_DIR}/{filename}", "wb") as file:
        pickle.dump(array, file)

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class DQNAgent():
    def __init__(self, env, bucket_size=20, replay_memory_size=50000):
        self.env = env
        self.obs_elements = len(self.env.observation_space.high)
        self.action_elements = self.env.action_space.n
        self.bucket_size =bucket_size

        self.episode_count = 0
        
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.model, self.target_model = self.CreateModels() # main model, target model

    def Train(self, terminal_state, update_target_model_per=5, min_replay_size=100, mini_batch_size=64, discount=0.95):

        if len(self.replay_memory) < min_replay_size:
            return
        
        mini_batch = random.sample(self.replay_memory, mini_batch_size)

        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs = self.model.predict(current_states, verbose=0)
        next_states = np.array([transition[3] for transition in mini_batch])
        future_qs = self.target_model.predict(next_states, verbose=0)

        X = []
        y = []
        for index, (current_state, action, reward, new_state, done) in enumerate(mini_batch):

            # Calculate new Q value
            if not done:
                max_future_q = np.max(future_qs[index])
                new_q = reward + discount * max_future_q
            else:
                new_q = reward

            current_q = current_qs[index]
            current_q[action] = new_q
            X.append(current_state)
            y.append(current_q)

        self.model.fit(np.array(X), np.array(y), batch_size=mini_batch_size, verbose=0, shuffle=False) # add tensorboard callback

        if terminal_state:
            self.episode_count += 1

        if self.episode_count % update_target_model_per == 0 and self.episode_count > 0:
            self.target_model.set_weights(self.model.get_weights())        

    def AppendReplayMemory(self, transition):
        self.replay_memory.append(transition)
        
    def GetQ(self, state):
        discrete_state = self.GetDiscreteState(state)
        return self.model.predict(np.array(discrete_state).reshape(-1, len(discrete_state)), verbose=0)[0]

    def GetObsWindowSize(self):
        discrete_os_size = [self.bucket_size] * self.obs_elements
        discrete_os_win_size = (self.env.observation_space.high - self.env.observation_space.low) / discrete_os_size
        return discrete_os_win_size

    def GetDiscreteState(self, state):
        discrete_state = (state - self.env.observation_space.low) / self.GetObsWindowSize()
        return discrete_state

    def CreateModels(self,number_of_nodes=64,lr=0.003,loss="mse", metrics="accuracy"):
        model = Sequential()
        model.add(Dense(number_of_nodes, input_shape=(self.obs_elements,)))
        model.add(Activation('relu'))

        model.add(Dense(number_of_nodes))
        model.add(Activation("relu"))

        model.add(Dense(self.action_elements))
        model.add(Activation("linear"))
        model.compile(loss=loss, optimizer=Adam(lr=lr), metrics=[metrics])

        target_model = Sequential()
        target_model.add(Dense(number_of_nodes, input_shape=(self.obs_elements,)))
        target_model.add(Activation('relu'))

        target_model.add(Dense(number_of_nodes))
        target_model.add(Activation("relu"))

        target_model.add(Dense(self.action_elements))
        target_model.add(Activation("linear"))
        target_model.compile(loss=loss, optimizer=Adam(lr=lr), metrics=[metrics])

        target_model.set_weights(model.get_weights())

        return model, target_model

env = gym.make("MountainCar-v0", render_mode = None)
OBS_SPACE_ELEMENTS = len(env.observation_space.high)
ACT_SPACE_ELEMENTS = env.action_space.n

SHOW_LOGS_EVERY = 5

SAVE_MODEL_EVERY = 100

agent = DQNAgent(env)

episode_rewards = []
episode_steps = []
episode_times = []

total_steps = 0
# Do training
t0 = 0
t1 = 0
t_in = time.time()
mean_reward = 0
for episode in range(1, EPISODES+1):

    
    print(f"------------------------------------------Episode {episode}-----------------------------------------------")
    print(f"--------------------------------Last Episode time {t1 - t0}------------------------------")
    
    done = False
    episode_reward = 0
    episode_step = 0
    state, _ = env.reset()
    state = agent.GetDiscreteState(state)
    t0 = time.time()
    while not done:
        #Take Action
        if np.random.random() > EPSILON:
            action = np.argmax(agent.GetQ(state))
        else: 
            action = random.randint(0, env.action_space.n-1)

        new_state, reward, termination, truncation, _ = env.step(action)
        done = termination or truncation
        new_state = agent.GetDiscreteState(new_state)

        episode_reward += reward
        
        transition = (state, action, reward, new_state, done)
        agent.AppendReplayMemory(transition)
        agent.Train(done)        

        state = new_state

        total_steps += 1

        if not done:
            episode_step += 1

    episode_rewards.append(episode_reward)
    episode_steps.append(episode_step)
    t1 = time.time()
    episode_times.append(t1 - t0)

    # Decay Epsilon
    if decay_start <= episode <= decay_end:
        EPSILON -= decay

    # Show logs
    if episode >= SHOW_LOGS_EVERY and episode % SHOW_LOGS_EVERY == 0:
        if episode >= 100:
            mean_reward = sum(episode_rewards[-100:]) / 100
            mean_episode_step = sum(episode_steps[-100:]) / 100
        max_reward = np.max(episode_rewards[-100:])
        min_reward = np.min(episode_rewards[-100:])
        print("x-------------------------------------x")
        print(f"Episode: {episode}")
        print(f"Total steps: {total_steps}")
        print(f"Max reward: {max_reward}")
        print(f"Min reward: {min_reward}")
        if episode >= 100:
            print(f"Mean reward: {mean_reward}")
            print(f"Mean steps per episode: {mean_episode_step}")
        print(f"Epsilon: {EPSILON}")
        print("x-------------------------------------x")

    # Save model
    if episode > 0 and episode % SAVE_MODEL_EVERY == 0 or episode==1:
         agent.target_model.save(f"{MODEL_DIR}/ep{episode}mean-rew-{mean_reward}")

t_tot = time.time()
################################################################################################################################
# Plot and save graphs
def moving_average(data, window_size):
    cumulative_sum = np.cumsum(data, dtype=float)
    cumulative_sum[window_size:] = cumulative_sum[window_size:] - cumulative_sum[:-window_size]
    return cumulative_sum[window_size - 1:] / window_size

MA_rewards = moving_average(episode_rewards, 100)
MA_steps = moving_average(episode_steps, 100)

# Save log data
SaveArray(episode_rewards, "rewards.pickle")
SaveArray(episode_times, "times.pickle")
SaveArray(episode_steps, "steps.pickle")

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(np.arange(100, EPISODES+1), MA_rewards, label='Moving Average of last 100 rewards')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Moving Average of Rewards Over Episodes')
plt.legend()
plt.grid(True)
plt.savefig(f"{LOGS_DIR}/mean_rewards.png")
plt.close() 

plt.figure(figsize=(10, 6))
plt.plot(np.arange(100, EPISODES+1), MA_steps, label='Moving Average of last 100 episode steps')
plt.xlabel('Episode')
plt.ylabel('Average number of steps')
plt.title('Moving Average of number of steps Per Episodes')
plt.legend()
plt.grid(True)
plt.savefig(f"{LOGS_DIR}/mean_steps.png")
plt.close() 