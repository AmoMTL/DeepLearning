from QLearn import QLearn
import gym

EPISODES_TO_RENDER = 5

env = gym.make("MountainCar-v0", render_mode="human")

Q = QLearn(env)

Q.load_qtable()

Q.render(EPISODES_TO_RENDER)