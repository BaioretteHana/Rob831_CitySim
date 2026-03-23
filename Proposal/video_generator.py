import gymnasium as gym
import highway_env

# env = gym.make(
#     "highway-v0",
#     render_mode="rgb_array"
# )

# env = gym.wrappers.RecordVideo(
#     env,
#     video_folder="videos",
#     episode_trigger=lambda e: True
# )

env_names = ["highway-v0", "merge-v0", "intersection-v0", "roundabout-v0"]

for i in range(4):

    env = gym.make(
        env_names[i],
        render_mode="rgb_array",
        config={
            "simulation_frequency": 15,
            "policy_frequency": 5
        },
        max_episode_steps=200
    )

    env = gym.wrappers.RecordVideo(
        env,
        video_folder= "videos",
        episode_trigger= lambda e: e == 0,
        name_prefix= env_names[i] + "_random_agent_sim"
    )

    obs, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()


    
