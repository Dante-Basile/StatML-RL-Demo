import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import imageio
import matplotlib.pyplot as plt
import numpy as np


train = True
ver = "2"


# Create environment
env_calm = gym.make("LunarLander-v2")
env_wind = gym.make("LunarLander-v2", enable_wind=True)

if train:

    # Instantiate the agent
    model_calm_wind = PPO("MlpPolicy", env_calm, verbose=1)
    # Train the agent and display a progress bar
    print("Training calm_wind on calm")
    model_calm_wind.learn(total_timesteps=int(2e5), progress_bar=True)
    model_calm_wind.set_env(env_wind)
    print("Training calm_wind on wind")
    model_calm_wind.learn(total_timesteps=int(2e5), progress_bar=True, reset_num_timesteps=False)

    model_wind_calm = PPO("MlpPolicy", env_wind, verbose=1)
    print("Training wind_calm on wind")
    model_wind_calm.learn(total_timesteps=int(2e5), progress_bar=True)
    model_wind_calm.set_env(env_calm)
    print("Training wind_calm on calm")
    model_wind_calm.learn(total_timesteps=int(2e5), progress_bar=True, reset_num_timesteps=False)

    # Save the agent
    model_calm_wind.save("ppo_lunar_primacy_calm_wind" + "_" + ver)
    model_wind_calm.save("ppo_lunar_primacy_wind_calm" + "_" + ver)
    del model_calm_wind  # delete trained model to demonstrate loading
    del model_wind_calm

# Load the trained agent
model_calm_wind = PPO.load("ppo_lunar_primacy_calm_wind" + "_" + ver, env=env_wind)
model_wind_calm = PPO.load("ppo_lunar_primacy_wind_calm" + "_" + ver, env=env_wind)

# Evaluate the agent
mean_reward_calm_wind, std_reward_calm_wind = evaluate_policy(model_calm_wind,
                                          model_calm_wind.get_env(),
                                          n_eval_episodes=100)
print(f"mean reward calm_wind: {mean_reward_calm_wind}")
print(f"std dev reward calm_wind: {std_reward_calm_wind}")

mean_reward_wind_calm, std_reward_wind_calm = evaluate_policy(model_wind_calm,
                                          model_wind_calm.get_env(),
                                          n_eval_episodes=100)

print(f"mean reward wind_calm: {mean_reward_wind_calm}")
print(f"std dev reward wind_calm: {std_reward_wind_calm}")

bar_x_calm_wind = [1]
bar_x_wind_calm = [2]
bar_y_calm_wind = [mean_reward_calm_wind]
bar_y_wind_calm = [mean_reward_wind_calm]
bar_err_calm_wind = [std_reward_calm_wind]
bar_err_wind_calm = [std_reward_wind_calm]
plt.bar(bar_x_calm_wind, bar_y_calm_wind, color='b')
plt.bar(bar_x_wind_calm, bar_y_wind_calm, color='g')
plt.errorbar(bar_x_calm_wind, bar_y_calm_wind, yerr=bar_err_calm_wind, fmt='o', color='r')
plt.errorbar(bar_x_wind_calm, bar_y_wind_calm, yerr=bar_err_wind_calm, fmt='o', color='r')
plt.legend(["calm->wind", "wind->calm"])
plt.xticks([])
plt.title("Agent Reward Comparison")
plt.ylabel("reward")
plt.savefig("ppo_lunar_primacy" + "_" + ver + ".png")

# Enjoy trained agent
env_disp = gym.make("LunarLander-v2", enable_wind=True, render_mode='rgb_array')
print("Demo calm_wind")
del model_calm_wind
model_calm_wind = PPO.load("ppo_lunar_primacy_calm_wind" + "_" + ver, env=env_disp)
vec_env = model_calm_wind.get_env()
obs = vec_env.reset()
frames_calm_wind = []
for i in range(1000):
    action, _states = model_calm_wind.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    frames_calm_wind.append(vec_env.render())
imageio.mimsave("ppo_lunar_primacy_calm_wind" + "_" + ver + ".gif",
                [np.array(img) for i, img in enumerate(frames_calm_wind) if i%2 == 0], fps=29)

print("Demo wind_calm")
del model_wind_calm
model_wind_calm = PPO.load("ppo_lunar_primacy_wind_calm" + "_" + ver, env=env_disp)
vec_env = model_wind_calm.get_env()
obs = vec_env.reset()
frames_wind_calm = []
for i in range(1000):
    action, _states = model_wind_calm.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    frames_wind_calm.append(vec_env.render())
imageio.mimsave("ppo_lunar_primacy_wind_calm" + "_" + ver + ".gif",
                [np.array(img) for i, img in enumerate(frames_wind_calm) if i%2 == 0], fps=29)
