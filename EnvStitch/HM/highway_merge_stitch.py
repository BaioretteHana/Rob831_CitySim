import sys
sys.path.insert(0, "/home/mugdha/coursework/IntroToRobLearning/Project/HighwayEnv")


import highway_env   # <-- add this
print(highway_env.__file__)

import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import imageio
from highway_env.envs.highway_stitchable_env import HighwayEnv
from highway_env.envs.merge_adapter_env import MergeAdapterEnv

frames = []

# Run highway
env = HighwayEnv(config={"screen_width": 1200, "screen_height": 1200, "duration": 40})
env.render_mode = "rgb_array"

obs, _ = env.reset()

done = False
while not done:
    # action = env.action_space.sample()
    action = 1
    obs, reward, terminated, truncated, _ = env.step(action)
    frames.append(env.render())
    done = terminated or truncated

# Extract ego state
ego = env.unwrapped.vehicle

handoff_state = {
    "speed": ego.speed,
    "lane": ego.lane_index[2],
    "position": ego.position[0],
    # "heading": ego.heading # Dont include heading
}

env.close()

# Run merge adapter
# merge_env = MergeAdapterEnv(config={"screen_width": 1200, "screen_height": 1200}, render_mode='rgb_array')
merge_env = MergeAdapterEnv(
    config={
        "screen_width": 1200,
        "screen_height": 1200,
        "duration": 60
    }
)
merge_env.render_mode = "rgb_array"
merge_env.set_handoff_state(handoff_state)

obs, _ = merge_env.reset()
frames.append(merge_env.render())

done = False
while not done:
    # action = merge_env.action_space.sample()
    action = 1
    obs, reward, terminated, truncated, _ = merge_env.step(action)
    frames.append(merge_env.render())
    done = terminated or truncated

merge_env.close()

# Saving the video
imageio.mimsave(
    "highway_to_merge.mp4",
    frames,
    fps=15
)

print("Saved video: highway_to_merge_stitched.mp4")