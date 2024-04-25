import gym
from QLearn import QLearn

EPSILON = 0.95
DECAY_START = 500
DECAY_END = 1500
EPISODES = 20_000

LEARNING_RATE = 0.5
DISCOUNT = 0.995

BUCKET_SIZE = 50

SHOW_LOGS_EVERY = 1000 # episodes

env = gym.make("MountainCar-v0", render_mode=None)

QLearnModel = QLearn(env, buckets=BUCKET_SIZE)

QLearnModel.Learn(epsilon=EPSILON, decay_start=DECAY_START, decay_end=DECAY_END, learning_rate=LEARNING_RATE, discount=DISCOUNT, episodes=EPISODES, logs_every=SHOW_LOGS_EVERY)

QLearnModel.plot_save_rewards_graph()

QLearnModel.save_best_model()