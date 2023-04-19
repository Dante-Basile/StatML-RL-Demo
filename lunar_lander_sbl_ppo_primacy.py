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
    print("Training calm_wind on calm")
    model_calm_wind.learn(total_timesteps=int(2e5), progress_bar=True)
    model_calm_wind.set_env(env_wind)
    print("Training calm_wind on wind")
    model_calm_wind.learn(total_timesteps=int(2e5), progress_bar=True)

    model_wind_calm = PPO("MlpPolicy", env_wind, verbose=1)
    print("Training wind_calm on wind")
    model_wind_calm.learn(total_timesteps=int(2e5), progress_bar=True)
    model_wind_calm.set_env(env_calm)
    print("Training wind_calm on calm")
    model_wind_calm.learn(total_timesteps=int(2e5), progress_bar=True)

    # Save the agent
    model_calm_wind.save("ppo_lunar_primacy_calm_wind")
    model_wind_calm.save("ppo_lunar_primacy_wind_calm")
    del model_calm_wind  # delete trained model to demonstrate loading
    del model_wind_calm

# Load the trained agent
model_calm_wind = PPO.load("ppo_lunar_primacy_calm_wind", env=env_wind)
model_wind_calm = PPO.load("ppo_lunar_primacy_wind_calm", env=env_wind)

# Evaluate the agent
mean_reward_calm_wind, std_reward_calm_wind = evaluate_policy(model_calm_wind,
                                          model_calm_wind.get_env(),
                                          n_eval_episodes=10)
mean_reward_wind_calm, std_reward_wind_calm = evaluate_policy(model_wind_calm,
                                          model_wind_calm.get_env(),
                                          n_eval_episodes=10)

print(f"mean reward calm_wind: {mean_reward_calm_wind}")
print(f"std dev reward calm_wind: {std_reward_calm_wind}")
print(f"mean reward wind_calm: {mean_reward_wind_calm}")
print(f"std dev reward wind_calm: {std_reward_wind_calm}")

# Enjoy trained agent
print("Demo calm_wind")
vec_env = model_calm_wind.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model_calm_wind.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()
time.sleep(2)
print("Demo wind_calm")
vec_env = model_wind_calm.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model_wind_calm.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()
