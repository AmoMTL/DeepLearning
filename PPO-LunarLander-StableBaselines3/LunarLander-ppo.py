import gymnasium as gym
from stable_baselines3 import A2C
import os
import time

env_name = "LunarLander-v2"
model_name = "A2C"

timesteps = 10_000

def CreateModelsLogDir(model_name="no name"):
    if model_name == "no name":
        model_dir = "models"
    else:
        model_dir = f"models/{model_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    logs_dir = "logs"
    return model_dir, logs_dir


def SaveModel(model, model_dir, model_name, timesteps, iter):  # for stable baselines models
    model.save(f"{model_dir}/{model_name}{timesteps*iter}")

model_dir, logs_dir = CreateModelsLogDir(model_name)

env = gym.make(env_name, render_mode=None)
env.reset()
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logs_dir)


iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=model_name)
    SaveModel(model, model_dir, model_name, timesteps, iters)




