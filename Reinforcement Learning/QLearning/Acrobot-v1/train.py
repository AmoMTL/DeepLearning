from QLearn import QLearn
import gym

EPSILON = 0.95
DECAY_START = 1000
DECAY_END = 30_000
EPISODES = 30_000

LEARNING_RATE = 0.1
DISCOUNT = 0.95

WIN = -100

BUCKET_SIZE = 20

SHOW_LOGS_EVERY = 100 # episodes

env_name = "Acrobot-v1"

QLearnModel = QLearn(env_name, buckets=BUCKET_SIZE)

QLearnModel.Learn(epsilon=EPSILON, decay_start=DECAY_START, decay_end=DECAY_END, learning_rate=LEARNING_RATE, discount=DISCOUNT, episodes=EPISODES, logs_every=SHOW_LOGS_EVERY, win=WIN)

QLearnModel.save_best_model()

QLearnModel.plot_save_rewards_graph()

QLearnModel.save_gif()