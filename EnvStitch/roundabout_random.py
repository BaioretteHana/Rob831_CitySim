import sys
sys.path.insert(0, "/home/mugdha/coursework/IntroToRobLearning/Project/HighwayEnv")


import highway_env   # <-- add this
print(highway_env.__file__)

import gymnasium as gym
from matplotlib import pyplot as plt
from highway_env.envs.roundabout_tester_env import RoundaboutEnv

env = RoundaboutEnv(config={"screen_width": 1200, "screen_height": 1200}, render_mode='rgb_array')
# env = MergeEnv(config={"lanes_count": 4}, render_mode='rgb_array')
# env.reset()

env = gym.wrappers.RecordVideo(
    env,
    video_folder="videos",
    episode_trigger=lambda e: e == 0,
    name_prefix="roundabout_test_v1_IDLE_sim"
)

obs, info = env.reset()
done = False

while not done:
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()

# for _ in range(25):
#     action = env.unwrapped.action_type.actions_indexes["IDLE"]
#     obs, reward, done, truncated, info = env.step(action)
#     env.render()

# plt.imshow(env.render())
# plt.show()