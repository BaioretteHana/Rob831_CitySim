import sys
sys.path.insert(0, "/home/mugdha/coursework/IntroToRobLearning/Project/HighwayEnv")


import highway_env   # <-- add this
print(highway_env.__file__)

import gymnasium as gym
from matplotlib import pyplot as plt
from highway_env.envs.merge_adapter_env import MergeAdapterEnv

env = MergeAdapterEnv(config={"lanes_count": 2, "screen_width": 1200, "screen_height": 1200}, render_mode='rgb_array')
# env = MergeEnv(config={"lanes_count": 4}, render_mode='rgb_array')

# Show one frame
obs, info = env.reset()
frame = env.render()
plt.imshow(frame)
plt.axis("off")
plt.show()

env.close()

plt.imshow(env.render())
plt.show()

# Video Gen
env = MergeAdapterEnv(config={"lanes_count": 2, "screen_width": 1200, "screen_height": 1200}, render_mode='rgb_array')

env = gym.wrappers.RecordVideo(
    env,
    video_folder= "videos",
    episode_trigger= lambda e: e == 0,
    name_prefix= "merge_4_v1" + "_IDLE_sim"
)

obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()

# from highway_env.envs.merge_spawn_env import MergeEnv

# env = MergeEnv(config={"lanes_count": 2, "screen_width": 1200, "screen_height": 1200}, render_mode='rgb_array')
# # env = MergeEnv(config={"lanes_count": 4}, render_mode='rgb_array')

# # Show one frame
# obs, info = env.reset()
# frame = env.render()
# plt.imshow(frame)
# plt.axis("off")
# plt.show()

# env.close()

# plt.imshow(env.render())
# plt.show()

# # Video Gen
# env = MergeEnv(config={"lanes_count": 2, "screen_width": 1200, "screen_height": 1200}, render_mode='rgb_array')

# env = gym.wrappers.RecordVideo(
#     env,
#     video_folder= "videos",
#     episode_trigger= lambda e: e == 0,
#     name_prefix= "merge_ego_spawn" + "_IDLE_sim"
# )

# obs, info = env.reset()
# done = False

# while not done:
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated

# env.close()

    
