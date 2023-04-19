import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


train = True


# Create environment
env_calm = gym.make("LunarLander-v2")
env_wind = gym.make("LunarLander-v2", enable_wind=True)

if train:

    # Instantiate the agent
    model_calm_wind = PPO("MlpPolicy", env_calm, verbose=1)
    # Train the agent and display a progress bar
    model_calm_wind.learn(total_timesteps=int(2e5), progress_bar=True)
    model_calm_wind.set_env(env_wind)
    model_calm_wind.learn(total_timesteps=int(2e5), progress_bar=True)

    model_wind_calm = PPO("MlpPolicy", env_wind, verbose=1)
    model_wind_calm.learn(total_timesteps=int(2e5), progress_bar=True)
    model_wind_calm.set_env(env_calm)
    model_wind_calm.learn(total_timesteps=int(2e5), progress_bar=True)

    # Save the agent
    model_calm_wind.save("ppo_lunar_primacy_calm_wind")
    del model_calm_wind  # delete trained model to demonstrate loading
    del model_wind_calm

# Load the trained agent
model = PPO.load("ppo_lunar_primacy", env=env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print(f"mean reward: {mean_reward}")
print(f"std dev reward: {std_reward}")

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()
