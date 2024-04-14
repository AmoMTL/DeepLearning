import gymnasium as gym
from stable_baselines3 import A2C

model_dir = "models"
model_type = "A2C"
model_names = ["A2C14120000"]
number_episodes = 4
env_name = "LunarLander-v2"

def RenderEpisode(env, model, episodes=1):
    for episode in range(episodes):
        
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs)
            new_state, reward, termination, truncation, _ = env.step(action)
            done = termination or truncation
            episode_reward += reward
            obs = new_state
        print(f"Episode: {episode+1} Reward: {episode_reward}")

env = gym.make(env_name, render_mode="human")

for model_name in model_names:
    model_path = f"{model_dir}\\{model_type}\\{model_name}"
    model = A2C.load(model_path, env)
    RenderEpisode(env, model, number_episodes)

env.close()  