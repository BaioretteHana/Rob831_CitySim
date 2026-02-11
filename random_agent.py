import gymnasium as gym
import highway_env
import numpy as np
from gymnasium.wrappers import TimeLimit

env_names = ["highway-v0", "merge-v0", "intersection-v0", "roundabout-v0"]

for i in range(4):

    env = gym.make(env_names[i]) # Will take a minute or so to run.
    # We can cap epi_length.
    # env = TimeLimit(gym.make("highway-v0"), max_episode_steps=200) # If running the random agent takes too long, since highway-v0 is expensive. 
    num_episodes = 50
    returns = []
    steps = []

    total_steps = 0

    for epi in range(num_episodes):
        obs, info = env.reset()
        done = False
        epi_return = 0
        epi_steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            epi_return += reward
            epi_steps += 1
            total_steps += 1

        returns.append(epi_return)
        steps.append(total_steps)

    env.close()

    random_returns = "random_returns_" + env_names[i] + ".npy"
    random_steps = "random_steps_" + env_names[i] + ".npy"
    np.save(random_returns, np.array(returns))
    np.save(random_steps, np.array(steps))
