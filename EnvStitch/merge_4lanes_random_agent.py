import sys
sys.path.insert(0, "/home/mugdha/coursework/IntroToRobLearning/Project/HighwayEnv")


import highway_env   # <-- add this
print(highway_env.__file__)


# import gymnasium
# from matplotlib import pyplot as plt
# from highway_env.envs.roundabout_env import RoundaboutEnv
# from highway_env.envs.roundabout_fixed_env import MergeEnv
# # import MergeRound

# env = MergeEnv(config={"lanes_count": 2, "screen_width": 1200, "screen_height": 1200}, render_mode='rgb_array')
# env.reset()

# for _ in range(25):
#     action = env.unwrapped.action_type.actions_indexes["IDLE"]
#     obs, reward, done, truncated, info = env.step(action)
#     env.render()

# plt.imshow(env.render())
# plt.show()

import gymnasium
from matplotlib import pyplot as plt
from highway_env.envs.merge_fixed_env import MergeAdapterEnv

env = MergeAdapterEnv(config={"lanes_count": 2, "screen_width": 1200, "screen_height": 1200}, render_mode='rgb_array')
env.reset()
# env = MergeEnv(config={"lanes_count": 4}, render_mode='rgb_array')
# env.reset()

for _ in range(5):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

plt.imshow(env.render())
plt.show()