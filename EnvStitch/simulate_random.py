# import os
# import sys
# sys.path.insert(0, "/home/mugdha/coursework/IntroToRobLearning/Project/HighwayEnv")

# import gymnasium
# from matplotlib import pyplot as plt
# from highway_env.envs.four_stitched_sqenly_envs import HMRIEnv

# env = HMRIEnv(config={"screen_width": 608, "screen_height": 608}, render_mode='rgb_array')
# obs, info = env.reset()


# for _ in range(40):
#     action = 1 #env.action_space.sample()
#     obs, reward, done, truncated, info = env.step(action)
#     env.render()

#     frame = env.render()

#     if done or truncated:
#         obs, info = env.reset()


import os
import sys
sys.path.insert(0, "/home/mugdha/coursework/IntroToRobLearning/Project/HighwayEnv")

import imageio
import numpy as np
from highway_env.envs.four_stitched_sqenly_envs import HMRIEnv

env = HMRIEnv(
    config={
        "screen_width": 608,
        "screen_height": 608,
        "offscreen_rendering": True,   # ← add this
    },
    render_mode="rgb_array"
)

obs, info = env.reset()
frames = []

for _ in range(150):
    action = 1
    obs, reward, done, truncated, info = env.step(action)
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    if done or truncated:
        obs, info = env.reset()

env.close()

os.makedirs("videos", exist_ok=True)
out_path = "videos/hmri_idle_rollout.mp4"
imageio.mimsave(out_path, frames, fps=15)
print(f"Saved {len(frames)} frames to {out_path}")