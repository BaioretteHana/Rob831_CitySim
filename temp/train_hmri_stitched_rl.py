import os
import sys
sys.path.insert(0, "/home/mugdha/coursework/IntroToRobLearning/Project/HighwayEnv")

import highway_env
import imageio
import argparse
import numpy as np
import gymnasium as gym
import pandas as pd

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.vec_env import DummyVecEnv

from highway_env.envs.four_stitched_sqenly_envs import HMRIEnv 

#  SUCC METRICS LOGGING
class MetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()

        # Per-episode metrics
        self.episode_collisions = 0
        self.episode_rewards = 0
        self.episode_length = 0

        self.episode_full_success = False
        self.episode_highway_success = False
        self.episode_merge_success = False
        self.episode_roundabout_success = False
        self.episode_intersection_success = False
        self.episode_merge_reached = False

        self.episode_highway_distance = 0
        self.episode_merge_distance = 0
        self.episode_roundabout_distance = 0
        self.episode_intersection_distance = 0

        self.episode_highway_speed = 0
        self.episode_merge_speed = 0
        self.episode_roundabout_speed = 0
        self.episode_intersection_speed = 0

        self.episode_step_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        done_flags = []

        for info in infos:
            self.episode_step_count += 1
            self.episode_rewards += info.get("reward", 0)

            # accumulate collisions
            self.episode_collisions += info.get("crashed", 0)

            # record success flags (overwrite if episode is done)
            self.episode_full_success = info.get("success_full", False)
            self.episode_highway_success = info.get("success_highway", False)
            self.episode_merge_success = info.get("success_merge", False)
            self.episode_roundabout_success = info.get("success_roundabout", False)
            self.episode_intersection_success = info.get("success_intersection", False)
            self.episode_merge_reached = info.get("merge_reached", False)

            # accumulate distance
            self.episode_highway_distance += info.get("highway_distance", 0)
            self.episode_merge_distance += info.get("merge_distance", 0)
            self.episode_roundabout_distance += info.get("roundabout_distance", 0)
            self.episode_intersection_distance += info.get("intersection_distance", 0)

            # accumulate speed for averaging
            self.episode_highway_speed += info.get("highway_speed", 0)
            self.episode_merge_speed += info.get("merge_speed", 0)
            self.episode_roundabout_speed += info.get("roundabout_speed", 0)
            self.episode_intersection_speed += info.get("intersection_speed", 0)

            done_flags.append(info.get("done", False) or info.get("truncated", False))

        # Check if episode ended
        if any(done_flags):
            steps = max(self.episode_step_count, 1)

            # Log success metrics
            self.logger.record("success/full", int(self.episode_full_success))
            self.logger.record("success/highway", int(self.episode_highway_success))
            self.logger.record("success/merge", int(self.episode_merge_success))
            self.logger.record("success/roundabout", int(self.episode_roundabout_success))
            self.logger.record("success/intersection", int(self.episode_intersection_success))
            self.logger.record("merge/reached_rate", int(self.episode_merge_reached))

            # Log distance metrics (total distance traveled this episode)
            self.logger.record("distance/highway", self.episode_highway_distance)
            self.logger.record("distance/merge", self.episode_merge_distance)
            self.logger.record("distance/roundabout", self.episode_roundabout_distance)
            self.logger.record("distance/intersection", self.episode_intersection_distance)

            # Log speed metrics (average speed this episode)
            self.logger.record("speed/highway", self.episode_highway_speed / steps)
            self.logger.record("speed/merge", self.episode_merge_speed / steps)
            self.logger.record("speed/roundabout", self.episode_roundabout_speed / steps)
            self.logger.record("speed/intersection", self.episode_intersection_speed / steps)

            # Log other metrics
            self.logger.record("collisions/episode", self.episode_collisions)
            self.logger.record("ep/length", self.episode_step_count)
            self.logger.record("ep/reward", self.episode_rewards)

            # Reset episode metrics
            self.episode_collisions = 0
            self.episode_rewards = 0
            self.episode_length = 0
            self.episode_full_success = False
            self.episode_highway_success = False
            self.episode_merge_success = False
            self.episode_roundabout_success = False
            self.episode_intersection_success = False
            self.episode_merge_reached = False
            self.episode_highway_distance = 0
            self.episode_merge_distance = 0
            self.episode_roundabout_distance = 0
            self.episode_intersection_distance = 0
            self.episode_highway_speed = 0
            self.episode_merge_speed = 0
            self.episode_roundabout_speed = 0
            self.episode_intersection_speed = 0
            self.episode_step_count = 0

        return True
        
    def _on_training_end(self) -> None:
        """
        Called at the end of training. Saves CSV.
        """
        if self.episode_data:
            df = pd.DataFrame(self.episode_data)
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            df.to_csv(self.save_path, index=False)
            print(f"[MetricsCallback] Saved metrics CSV to {self.save_path}")
# class MetricsCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super().__init__(verbose)
#         # store recent 100-step histories
#         self.full_success = []
#         self.highway_success = []
#         self.merge_success = []
#         self.roundabout_success = []
#         self.intersection_success = []
#         self.merge_reached = []

#     def _on_step(self) -> bool:
#         infos = self.locals.get("infos", [])

#         for info in infos:
#             # append if key exists, else append 0 (or False)
#             self.full_success.append(info.get("success_full", 0))
#             self.highway_success.append(info.get("success_highway", 0))
#             self.merge_success.append(info.get("success_merge", 0))
#             self.roundabout_success.append(info.get("success_roundabout", 0))
#             self.intersection_success.append(info.get("success_intersection", 0))
#             self.merge_reached.append(info.get("merge_reached", 0))

#         # compute running mean over last 100 steps (or less if starting)
#         def mean_last(lst):
#             return np.mean(lst[-100:]) if lst else 0

#         self.logger.record("success/full", mean_last(self.full_success))
#         self.logger.record("success/highway", mean_last(self.highway_success))
#         self.logger.record("success/merge", mean_last(self.merge_success))
#         self.logger.record("success/roundabout", mean_last(self.roundabout_success))
#         self.logger.record("success/intersection", mean_last(self.intersection_success))
#         self.logger.record("merge/reached_rate", mean_last(self.merge_reached))

#         # # flush logger every step to ensure TensorBoard sees it immediately
#         # self.logger.dump(self.num_timesteps)

#         return True


# RANDOM AGENT
def run_random(env, steps, logger=None):
    returns = []
    successes = []

    obs, _ = env.reset()
    total = 0
    episode = 0

    for step in range(steps):
        action = env.action_space.sample()
        obs, r, done, trunc, info = env.step(action)
        total += r

        if done or trunc:
            returns.append(total)
            success = info.get("success_full", 0)
            successes.append(success)

            if logger:
                logger.record("random/return", total)
                logger.record("random/success", success)
                logger.record("random/episode", episode)
                logger.dump(step)

            obs, _ = env.reset()
            total = 0
            episode += 1

    return returns, successes


# VIDEO RECORDING
def record_final_rollout(model, config, video_name):
    env = HMRIEnv(config)
    env.render_mode = "rgb_array"

    obs, _ = env.reset()
    frames = []
    done = False

    while not done:
        for _ in range(2):  # frame repeat for smoothness
            frame = env.render()
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    os.makedirs("videos", exist_ok=True)
    imageio.mimsave(f"videos/{video_name}.mp4", frames, fps=20)



# MAIN TRAINING FUNCTION
def main(args):

    # ENV CONFIGURATION
    base_config = {
        "duration": args.ep_length,
        "simulation_frequency": 5,
        "policy_frequency": 1,
        "vehicles_count": 20,
        "screen_width": args.screen_width,
        "screen_height": args.screen_height,
        "offscreen_rendering": True,  # keep offscreen for training speed
        "render_agent": True,
        "show_trajectories": False,
    }

    def make_env():
        return Monitor(HMRIEnv(base_config, render_mode=None))

    env = DummyVecEnv([make_env])

    # env = VecVideoRecorder(
    #     env,
    #     video_folder="./videos/",
    #     record_video_trigger=lambda step: step % 50000 == 0,
    #     video_length=args.ep_length,
    #     name_prefix=f"{args.algo}_hmri"
    # )

    # SELECT ALGO
    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            verbose=1,
            n_steps=args.n_steps,
            tensorboard_log="./tensorboard/hmri/",
            device=args.device
        )
    elif args.algo == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            verbose=1,
            tensorboard_log="./tensorboard/hmri/",
            device=args.device
        )
    elif args.algo == "random":
        from stable_baselines3.common.logger import configure
        logger = configure("./tensorboard/random/", ["tensorboard"])
        env = HMRIEnv(base_config)
        run_random(env, args.timesteps, logger)
        return

    # TRAIN
    callback = MetricsCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(f"{args.algo}_hmri_model")

    # FINAL VIDEO
    record_final_rollout(model, base_config, f"{args.algo}_hmri_final")


# ARGUMENT PARSER

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn", "random"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=150000)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ep_length", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--screen_width", type=int, default=608)
    parser.add_argument("--screen_height", type=int, default=608)
    args = parser.parse_args()

    main(args)