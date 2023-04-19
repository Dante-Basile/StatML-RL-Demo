import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import imageio
import matplotlib.pyplot as plt 
import numpy as np


train = False


# Create environment
env = gym.make("LunarLander-v2", enable_wind=True)

if train:
    # Instantiate the agent
    model = PPO("MlpPolicy", env, verbose=1)
    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(2e5), progress_bar=True)
    # Save the agent
    model.save("ppo_lunar_wind")
    del model  # delete trained model to demonstrate loading

# Load the trained agent
model = PPO.load("ppo_lunar_wind", env=env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"mean reward: {mean_reward}")
print(f"std dev reward: {std_reward}")
bar_x = [1]
bar_y = [mean_reward]
bar_err = [std_reward]
plt.bar(bar_x, bar_y)
plt.errorbar(bar_x, bar_y, yerr=bar_err, fmt='o', color='r')
plt.savefig("ppo_lunar_wind.png")

# Enjoy trained agent
env_disp = gym.make("LunarLander-v2", enable_wind=True, render_mode='rgb_array')
del model
model = PPO.load("ppo_lunar_wind", env=env_disp)
vec_env = model.get_env()
obs = vec_env.reset()
frames = []
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    frames.append(vec_env.render())
imageio.mimsave("ppo_lunar_wind.gif",
                [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=29)
