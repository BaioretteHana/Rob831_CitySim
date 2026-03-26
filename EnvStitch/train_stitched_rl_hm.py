
import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"

import sys
sys.path.insert(0, "/home/mugdha/coursework/IntroToRobLearning/Project/HighwayEnv")

import highway_env   # <-- add this
print(highway_env.__file__)

import imageio
import argparse
import numpy as np
import gymnasium as gym
import pandas as pd

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from highway_env.envs.highway_stitchable_env import HighwayEnv
from highway_env.envs.merge_adapter_env import MergeAdapterEnv


# STITCHED ENV
class HighwayMergeEnv(gym.Env):

    def __init__(self, config):
        super().__init__()

        self.highway = HighwayEnv(config=config)
        self.merge = MergeAdapterEnv(config=config)

        # self.highway.render_mode = None
        # self.merge.render_mode = None

        self.render_mode = "rgb_array"

        self.highway.render_mode = "rgb_array"
        self.merge.render_mode = "rgb_array"

        self.action_space = self.highway.action_space
        self.observation_space = self.highway.observation_space

        self.phase = "highway"
        self.reached_merge = False

    def reset(self, seed=None, options=None):
        self.phase = "highway"
        self.reached_merge = False
        obs, info = self.highway.reset()
        return obs, info

    def render(self):
        if self.phase == "highway":
            return self.highway.render()
        else:
            return self.merge.render()
    
    def _normalize_reward(self, reward, phase):
        if phase == "highway":
            # approximate min/max
            highway_min, highway_max = -1.05, 0.5
            reward = (reward - highway_min) / (highway_max - highway_min)
        else:
            # merge env
            merge_min, merge_max = -1.55, 0.3
            reward = (reward - merge_min) / (merge_max - merge_min)
        return reward

    def step(self, action):

        # HIGHWAY 
        if self.phase == "highway":
            obs, reward, terminated, truncated, info = self.highway.step(action)
            reward = self._normalize_reward(reward, self.phase)

            if terminated or truncated:

                # crash -> total failure
                if self.highway.vehicle.crashed:
                    info["success_highway"] = False
                    info["success_merge"] = False
                    info["success_full"] = False
                    info["merge_reached"] = False
                    return obs, reward, True, truncated, info

                # reached end -> go to merge
                ego = self.highway.vehicle
                handoff = {
                    "speed": ego.speed,
                    "lane": ego.lane_index[2],
                    "position": ego.position[0],
                }

                self.merge.set_handoff_state(handoff)
                obs, _ = self.merge.reset()

                self.phase = "merge"
                self.reached_merge = True

                return obs, reward, False, False, info

            return obs, reward, False, False, info

        # MERGE
        else:
            obs, reward, terminated, truncated, info = self.merge.step(action)
            reward = self._normalize_reward(reward, self.phase)

            if terminated or truncated:

                crashed = self.merge.vehicle.crashed

                info["success_highway"] = True
                info["success_merge"] = not crashed
                info["success_full"] = not crashed
                info["merge_reached"] = True

            return obs, reward, terminated, truncated, info


# LOGGING
class MetricsCallback(BaseCallback):

    def __init__(self):
        super().__init__()
        self.full_success = []
        self.highway_success = []
        self.merge_success = []
        self.merge_reached = []

    def _on_step(self):

        infos = self.locals.get("infos", [])

        for info in infos:

            if "success_full" in info:
                self.full_success.append(info["success_full"])
                self.highway_success.append(info["success_highway"])
                self.merge_success.append(info["success_merge"])
                self.merge_reached.append(info["merge_reached"])

        if len(self.full_success) > 0:
            self.logger.record("success/full", np.mean(self.full_success[-100:]))
            self.logger.record("success/highway", np.mean(self.highway_success[-100:]))
            self.logger.record("success/merge", np.mean(self.merge_success[-100:]))
            self.logger.record("merge/reached_rate", np.mean(self.merge_reached[-100:]))

        return True

# RANDOM AGENT
def run_random(env, steps):

    returns = []
    successes = []

    obs,_ = env.reset()
    total = 0

    for _ in range(steps):

        action = env.action_space.sample()
        obs, r, done, trunc, info = env.step(action)
        total += r

        if done or trunc:
            returns.append(total)
            successes.append(info.get("success_full", 0))
            obs,_ = env.reset()
            total = 0

    return returns, successes

def record_final_rollout(model, config, video_name):

    env = HighwayMergeEnv(config)

    env.highway.render_mode = "rgb_array"
    env.merge.render_mode = "rgb_array"

    obs, _ = env.reset()
    frames = []

    done = False
    while not done:

        # frame = env.render()
        # frames.append(frame)
        for _ in range(2):   # increase smoothness- Render Repeat!
            frame = env.render()
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

    os.makedirs("videos", exist_ok=True)
    imageio.mimsave(f"videos/{video_name}.mp4", frames, fps=20) # Pretty jumpy for 5 

# MAIN
def main(args):

    # config = {
    #     "duration": args.ep_length
    # }
    config = {
        "duration": args.ep_length,

        # ---- SPEED OPTIMIZATION ----
        "simulation_frequency": 5,      # lower physics updates (default ~15)
        "policy_frequency": 1,

        # fewer vehicles (big speed boost)
        "vehicles_count": 20,

        # smaller rendering buffer even if not rendering
        "screen_width": 600,
        "screen_height": 600,

        # disable heavy logging inside highway-env
        "show_trajectories": False,

        # causes lag and has to kill the process sometimes :(
        "offscreen_rendering": True,
        "render_agent": True
    }

    # env = Monitor(HighwayMergeEnv(config))
    def make_env():
        return Monitor(HighwayMergeEnv(config))

    env = DummyVecEnv([make_env])

    env = VecVideoRecorder(
        env,
        video_folder="./videos/",
        record_video_trigger=lambda step: step % 50000 == 0,
        video_length=args.ep_length,
        name_prefix= args.algo + "_stitched"
    )

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            verbose=1,
            n_steps=args.n_steps,
            tensorboard_log="./tensorboard/",
            device=args.device
        )

    elif args.algo == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            verbose=1,
            tensorboard_log="./tensorboard/",
            device=args.device
        )

    callback = MetricsCallback()

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback
    )

    model.save(f"{args.algo}_model")
    # record one final rollout
    record_final_rollout(model, config, f"{args.algo}_final_success")
    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", choices=["ppo","dqn","random"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=150000) # total training steps
    parser.add_argument("--n_steps", type=int, default=512) # how often PPO updates
    parser.add_argument("--batch_size", type=int, default=64) # minibatch size for gradient updates
    parser.add_argument("--ep_length", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()

    # if args.algo == "random":
    #     env = HighwayMergeEnv({"duration": args.ep_length})
    #     returns, success = run_random(env, args.timesteps)

    #     pd.DataFrame({
    #         "return": returns,
    #         "success": success
    #     }).to_csv("random_results.csv", index=False)

    if args.algo == "random":

        from stable_baselines3.common.logger import configure

        log_path = "./tensorboard/random/"
        logger = configure(log_path, ["tensorboard"])

        env = HighwayMergeEnv({"duration": args.ep_length})

        returns = []
        successes = []

        obs, _ = env.reset()
        total = 0
        episode = 0

        for step in range(args.timesteps):

            action = env.action_space.sample()
            obs, r, done, trunc, info = env.step(action)
            total += r

            if done or trunc:
                returns.append(total)
                success = info.get("success_full", 0)
                successes.append(success)

                logger.record("random/return", total)
                logger.record("random/success", success)
                logger.record("random/episode", episode)
                logger.dump(step)

                obs, _ = env.reset()
                total = 0
                episode += 1

        pd.DataFrame({
            "return": returns,
            "success": successes
        }).to_csv("random_results.csv", index=False)

    else:
        main(args)