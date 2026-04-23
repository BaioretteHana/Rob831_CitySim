"""
train_hmri_stitched_rl.py

Trains PPO, DQN, or a random agent on the stitched HMRI environment
(Highway → MergeAdapter → Merge → Roundabout → Intersection).

Logs to TensorBoard:
  - Learning curve (mean return vs timesteps)          ← required by midterm
  - Segment reached per episode                        ← progress / success metric
  - Episode length                                     ← survival proxy
  - Per-segment distance and speed                     ← granular analysis

Usage:
  python train_hmri_stitched_rl.py --algo ppo --timesteps 150000
  python train_hmri_stitched_rl.py --algo dqn --timesteps 150000
  python train_hmri_stitched_rl.py --algo random --timesteps 20000
"""

from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import imageio
import gymnasium as gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

sys.path.insert(0, "/home/mugdha/coursework/IntroToRobLearning/Project/HighwayEnv")
import highway_env
from highway_env.envs.four_stitched_sqenly_envs import HMRIEnv


# ── Segment encoding ─────────────────────────────────────────────────────────
# Used to turn "how far did the agent get" into a single integer for logging.
SEGMENT_ORDER = ["highway", "merge", "roundabout", "intersection"]
SEGMENT_RANK  = {seg: i for i, seg in enumerate(SEGMENT_ORDER)}


# ─────────────────────────────────────────────────────────────────────────────
# ENV FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_env_fn(config: dict, render_mode=None):
    """Returns a callable that creates a monitored HMRIEnv."""
    def _init():
        env = HMRIEnv(render_mode=render_mode)
        env.config.update(config)
        env = Monitor(env)
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# METRICS CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class HMRIMetricsCallback(BaseCallback):
    """
    Logs per-episode metrics to TensorBoard at episode boundaries.

    SB3 note: inside _on_step, per-step info comes from self.locals["infos"]
    (a list, one entry per env). Episode done signals come from
    self.locals["dones"]. Rewards come from self.locals["rewards"].
    Never read reward from info dict — Monitor wraps it separately.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._reset_episode()

    def _reset_episode(self):
        self.ep_reward        = 0.0
        self.ep_length        = 0
        self.ep_segment_rank  = 0          # highest segment reached this episode
        self.ep_last_segment  = "highway"

        # Per-segment accumulators (distance in metres, speed in m/s steps)
        self.seg_distance = {s: 0.0 for s in SEGMENT_ORDER}
        self.seg_speed    = {s: 0.0 for s in SEGMENT_ORDER}
        self.seg_steps    = {s: 0   for s in SEGMENT_ORDER}

    def _on_step(self) -> bool:
        # SB3 provides these as lists (one per parallel env — we use 1 env)
        infos   = self.locals["infos"]
        dones   = self.locals["dones"]
        rewards = self.locals["rewards"]

        # ── Read speed and segment directly from the unwrapped env ────────────
        # This bypasses the info dict entirely, which may be empty or stripped
        # by the Monitor wrapper depending on the SB3/gymnasium version.
        # Unwrap chain: DummyVecEnv → Monitor → HMRIEnv
        try:
            raw_env   = self.training_env.envs[0].env   # unwrap Monitor
            ego_speed = float(raw_env.vehicle.speed) if raw_env.vehicle else 0.0
            dt        = 1.0 / raw_env.config.get("policy_frequency", 1)
            step_dist = ego_speed * dt
            seg       = raw_env._get_segment()
        except Exception:
            # Fallback to info dict if unwrap fails for any reason
            ego_speed = float(infos[0].get("speed", 0.0)) if infos else 0.0
            step_dist = float(infos[0].get("step_distance", 0.0)) if infos else 0.0
            seg       = infos[0].get("segment", "highway") if infos else "highway"

        if seg not in SEGMENT_RANK:
            seg = "highway"

        for info, done, reward in zip(infos, dones, rewards):
            self.ep_reward += float(reward)
            self.ep_length += 1

            # ── Per-step accumulation ─────────────────────────────────────
            # Track furthest segment reached
            if SEGMENT_RANK[seg] > self.ep_segment_rank:
                self.ep_segment_rank = SEGMENT_RANK[seg]
            self.ep_last_segment = seg

            self.seg_distance[seg] += step_dist
            self.seg_speed[seg]    += ego_speed
            self.seg_steps[seg]    += 1

            # ── Episode boundary ──────────────────────────────────────────
            if done:
                full_succ = bool(info.get("success_full", False))

                # ── Success / progress ────────────────────────────────────
                self.logger.record("success/segment_reached", self.ep_segment_rank)
                self.logger.record("success/full_completion", int(full_succ))

                # Binary reached-each-segment flags
                for seg_name, rank in SEGMENT_RANK.items():
                    self.logger.record(
                        f"success/reached_{seg_name}",
                        int(self.ep_segment_rank >= rank)
                    )

                # ── Episode summary ───────────────────────────────────────
                self.logger.record("episode/mean_return", self.ep_reward)
                self.logger.record("episode/length",      self.ep_length)

                # ── Per-segment distance and avg speed ────────────────────
                for seg_name in SEGMENT_ORDER:
                    self.logger.record(
                        f"distance/{seg_name}",
                        self.seg_distance[seg_name]
                    )
                    steps = max(self.seg_steps[seg_name], 1)
                    self.logger.record(
                        f"speed/{seg_name}",
                        self.seg_speed[seg_name] / steps
                    )

                self.logger.dump(self.num_timesteps)
                self._reset_episode()

        return True


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO CALLBACK  (fires every N timesteps via EveryNTimesteps wrapper)
# ─────────────────────────────────────────────────────────────────────────────

class VideoRecorderCallback(BaseCallback):
    """
    Records a single deterministic rollout and saves it as an MP4.
    Wrap with EveryNTimesteps to fire periodically.
    """

    def __init__(self, config: dict, video_dir: str = "videos", verbose: int = 0):
        super().__init__(verbose)
        self.config    = config
        self.video_dir = video_dir
        os.makedirs(video_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Triggered by EveryNTimesteps — record one rollout
        env = HMRIEnv(render_mode="rgb_array")
        env.config.update(self.config)

        obs, _ = env.reset()
        frames = []
        done   = False

        while not done:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        env.close()

        if frames:
            path = os.path.join(
                self.video_dir,
                f"rollout_{self.num_timesteps:08d}.mp4"
            )
            imageio.mimsave(path, frames, fps=15)
            if self.verbose:
                print(f"[Video] Saved {path}")

        return True


# ─────────────────────────────────────────────────────────────────────────────
# RANDOM AGENT
# ─────────────────────────────────────────────────────────────────────────────

def run_random_agent(config: dict, timesteps: int, log_dir: str):
    """
    Rolls out a random agent, logging the same metrics as the trained agents
    so the midterm overlay plot has a consistent baseline.
    Returns (mean_return, std_return) over all completed episodes.
    """
    logger = configure(log_dir, ["tensorboard", "stdout"])

    env = HMRIEnv(render_mode=None)
    env.config.update(config)

    episode_returns   = []
    episode_lengths   = []
    segment_ranks     = []

    obs, _      = env.reset()
    ep_reward   = 0.0
    ep_length   = 0
    ep_seg_rank = 0
    step        = 0

    while step < timesteps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        ep_reward   += reward
        ep_length   += 1
        seg          = info.get("segment", "highway")
        rank         = SEGMENT_RANK.get(seg, 0)
        if rank > ep_seg_rank:
            ep_seg_rank = rank

        step += 1

        if done:
            episode_returns.append(ep_reward)
            episode_lengths.append(ep_length)
            segment_ranks.append(ep_seg_rank)

            logger.record("random/mean_return",     ep_reward)
            logger.record("random/episode_length",  ep_length)
            logger.record("random/segment_reached", ep_seg_rank)
            logger.dump(step)

            obs, _      = env.reset()
            ep_reward   = 0.0
            ep_length   = 0
            ep_seg_rank = 0

    env.close()

    mean_r = float(np.mean(episode_returns)) if episode_returns else 0.0
    std_r  = float(np.std(episode_returns))  if episode_returns else 0.0
    print(f"\n[Random Agent] Episodes: {len(episode_returns)}")
    print(f"  Mean return:      {mean_r:.3f} ± {std_r:.3f}")
    print(f"  Mean seg reached: {np.mean(segment_ranks):.2f}")
    return mean_r, std_r


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args):

    base_config = {
        "duration":             args.ep_length,
        "simulation_frequency": 5,
        "policy_frequency":     1,
        "vehicles_count":       0,       # start with 0 — add NPCs when ready
        "screen_width":         args.screen_width,
        "screen_height":        args.screen_height,
        "offscreen_rendering":  True,
        "render_agent":         True,
        "show_trajectories":    False,
    }

    tb_log_dir = f"./tensorboard/hmri_{args.algo}/"
    video_dir  = f"./videos/{args.algo}/"
    model_path = f"{args.algo}_hmri_model"

    # ── Random agent ─────────────────────────────────────────────────────────
    if args.algo == "random":
        run_random_agent(base_config, args.timesteps, "./tensorboard/hmri_random/")
        return

    # ── Vectorised training env ───────────────────────────────────────────────
    env = DummyVecEnv([make_env_fn(base_config)])

    # ── Model ─────────────────────────────────────────────────────────────────
    shared_kwargs = dict(
        env             = env,
        verbose         = 1,
        tensorboard_log = tb_log_dir,
        device          = args.device,
    )

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            learning_rate = args.lr,
            n_steps       = args.n_steps,
            batch_size    = args.batch_size,
            n_epochs      = 10,
            ent_coef      = 0.01,
            **shared_kwargs,
        )

    elif args.algo == "dqn":
        model = DQN(
            "MlpPolicy",
            learning_rate          = args.lr,
            batch_size             = args.batch_size,
            buffer_size            = 50_000,
            learning_starts        = 1_000,
            target_update_interval = 500,
            exploration_fraction   = 0.3,
            exploration_final_eps  = 0.05,
            **shared_kwargs,
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    metrics_cb = HMRIMetricsCallback(verbose=1)

    video_cb = EveryNTimesteps(
        n_steps  = args.video_interval,
        callback = VideoRecorderCallback(base_config, video_dir=video_dir, verbose=1),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps = args.timesteps,
        callback        = [metrics_cb, video_cb],
        tb_log_name     = args.algo,
    )

    model.save(model_path)
    print(f"\n[Training] Model saved to {model_path}.zip")

    # ── Final video ───────────────────────────────────────────────────────────
    print("[Training] Recording final rollout...")
    final_cb = VideoRecorderCallback(base_config, video_dir=video_dir, verbose=1)
    final_cb.num_timesteps = args.timesteps
    final_cb.model = model
    final_cb._on_step()


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL on stitched HMRI env")

    parser.add_argument("--algo",           choices=["ppo", "dqn", "random"], default="ppo")
    parser.add_argument("--timesteps",      type=int,   default=150_000)
    parser.add_argument("--ep_length",      type=int,   default=200,
                        help="Max steps per episode (duration in env config)")
    parser.add_argument("--n_steps",        type=int,   default=1024,
                        help="PPO: steps collected per update")
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--device",         type=str,   default="auto")
    parser.add_argument("--screen_width",   type=int,   default=608)
    parser.add_argument("--screen_height",  type=int,   default=608)
    parser.add_argument("--video_interval", type=int,   default=25_000,
                        help="Record a video every N timesteps")

    args = parser.parse_args()
    main(args)

# """
# train_hmri_stitched_rl.py

# Trains PPO, DQN, or a random agent on the stitched HMRI environment
# (Highway → MergeAdapter → Merge → Roundabout → Intersection).

# Logs to TensorBoard:
#   - Learning curve (mean return vs timesteps)          ← required by midterm
#   - Segment reached per episode                        ← progress / success metric
#   - Collision rate                                     ← failure case analysis
#   - Episode length                                     ← survival proxy
#   - Per-segment distance and speed                     ← granular analysis

# Usage:
#   python train_hmri_stitched_rl.py --algo ppo --timesteps 150000
#   python train_hmri_stitched_rl.py --algo dqn --timesteps 150000
#   python train_hmri_stitched_rl.py --algo random --timesteps 20000
# """

# from __future__ import annotations

# import os
# import sys
# import argparse

# import numpy as np
# import imageio
# import gymnasium as gym

# from stable_baselines3 import PPO, DQN
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.logger import configure

# sys.path.insert(0, "/home/mugdha/coursework/IntroToRobLearning/Project/HighwayEnv")
# import highway_env
# from highway_env.envs.four_stitched_sqenly_envs import HMRIEnv


# # ── Segment encoding ─────────────────────────────────────────────────────────
# # Used to turn "how far did the agent get" into a single integer for logging.
# SEGMENT_ORDER = ["highway", "merge", "roundabout", "intersection"]
# SEGMENT_RANK  = {seg: i for i, seg in enumerate(SEGMENT_ORDER)}


# # ─────────────────────────────────────────────────────────────────────────────
# # ENV FACTORY
# # ─────────────────────────────────────────────────────────────────────────────

# def make_env_fn(config: dict, render_mode=None):
#     """Returns a callable that creates a monitored HMRIEnv."""
#     def _init():
#         env = HMRIEnv(render_mode=render_mode)
#         env.config.update(config)
#         env = Monitor(env)
#         return env
#     return _init


# # ─────────────────────────────────────────────────────────────────────────────
# # METRICS CALLBACK
# # ─────────────────────────────────────────────────────────────────────────────

# class HMRIMetricsCallback(BaseCallback):
#     """
#     Logs per-episode metrics to TensorBoard at episode boundaries.

#     SB3 note: inside _on_step, per-step info comes from self.locals["infos"]
#     (a list, one entry per env). Episode done signals come from
#     self.locals["dones"]. Rewards come from self.locals["rewards"].
#     Never read reward from info dict — Monitor wraps it separately.
#     """

#     def __init__(self, verbose: int = 0):
#         super().__init__(verbose)
#         self._reset_episode()

#     def _reset_episode(self):
#         self.ep_reward        = 0.0
#         self.ep_length        = 0
#         self.ep_collisions    = 0
#         self.ep_segment_rank  = 0          # highest segment reached this episode
#         self.ep_last_segment  = "highway"

#         # Per-segment accumulators (distance in metres, speed in m/s steps)
#         self.seg_distance = {s: 0.0 for s in SEGMENT_ORDER}
#         self.seg_speed    = {s: 0.0 for s in SEGMENT_ORDER}
#         self.seg_steps    = {s: 0   for s in SEGMENT_ORDER}

#     def _on_step(self) -> bool:
#         # SB3 provides these as lists (one per parallel env — we use 1 env)
#         infos   = self.locals["infos"]
#         dones   = self.locals["dones"]
#         rewards = self.locals["rewards"]

#         # ── Read speed and segment directly from the unwrapped env ────────────
#         # This bypasses the info dict entirely, which may be empty or stripped
#         # by the Monitor wrapper depending on the SB3/gymnasium version.
#         # Unwrap chain: DummyVecEnv → Monitor → HMRIEnv
#         try:
#             raw_env   = self.training_env.envs[0].env   # unwrap Monitor
#             ego_speed = float(raw_env.vehicle.speed) if raw_env.vehicle else 0.0
#             dt        = 1.0 / raw_env.config.get("policy_frequency", 1)
#             step_dist = ego_speed * dt
#             seg       = raw_env._get_segment()
#             crashed_direct = bool(raw_env.vehicle.crashed) if raw_env.vehicle else False
#         except Exception:
#             # Fallback to info dict if unwrap fails for any reason
#             ego_speed      = float(infos[0].get("speed", 0.0)) if infos else 0.0
#             step_dist      = float(infos[0].get("step_distance", 0.0)) if infos else 0.0
#             seg            = infos[0].get("segment", "highway") if infos else "highway"
#             crashed_direct = bool(infos[0].get("crashed", False)) if infos else False

#         if seg not in SEGMENT_RANK:
#             seg = "highway"

#         for info, done, reward in zip(infos, dones, rewards):
#             self.ep_reward += float(reward)
#             self.ep_length += 1

#             # ── Per-step accumulation ─────────────────────────────────────
#             # Track furthest segment reached
#             if SEGMENT_RANK[seg] > self.ep_segment_rank:
#                 self.ep_segment_rank = SEGMENT_RANK[seg]
#             self.ep_last_segment = seg

#             self.ep_collisions  += int(crashed_direct)
#             self.seg_distance[seg] += step_dist
#             self.seg_speed[seg]    += ego_speed
#             self.seg_steps[seg]    += 1

#             # ── Episode boundary ──────────────────────────────────────────
#             if done:
#                 crashed   = crashed_direct
#                 full_succ = bool(info.get("success_full", False))

#                 # ── Success / progress ────────────────────────────────────
#                 self.logger.record("success/segment_reached",   self.ep_segment_rank)
#                 self.logger.record("success/full_completion",   int(full_succ))

#                 # Binary reached-each-segment flags
#                 for seg_name, rank in SEGMENT_RANK.items():
#                     self.logger.record(
#                         f"success/reached_{seg_name}",
#                         int(self.ep_segment_rank >= rank)
#                     )

#                 # ── Episode summary ───────────────────────────────────────
#                 self.logger.record("episode/mean_return", self.ep_reward)
#                 self.logger.record("episode/length",      self.ep_length)

#                 # ── Collision / failure ───────────────────────────────────
#                 self.logger.record("failure/crashed", int(crashed))
#                 self.logger.record("failure/crash_in_segment",  self.ep_segment_rank
#                                    if crashed else -1)
#                 # Timeout = truncated without crash
#                 timed_out = (not crashed) and (not full_succ)
#                 self.logger.record("failure/timed_out", int(timed_out))

#                 # ── Per-segment distance and avg speed ────────────────────
#                 for seg_name in SEGMENT_ORDER:
#                     self.logger.record(
#                         f"distance/{seg_name}",
#                         self.seg_distance[seg_name]
#                     )
#                     steps = max(self.seg_steps[seg_name], 1)
#                     self.logger.record(
#                         f"speed/{seg_name}",
#                         self.seg_speed[seg_name] / steps
#                     )

#                 self.logger.dump(self.num_timesteps)
#                 self._reset_episode()

#         return True


# # ─────────────────────────────────────────────────────────────────────────────
# # VIDEO CALLBACK  (fires every N timesteps via EveryNTimesteps wrapper)
# # ─────────────────────────────────────────────────────────────────────────────

# class VideoRecorderCallback(BaseCallback):
#     """
#     Records a single deterministic rollout and saves it as an MP4.
#     Wrap with EveryNTimesteps to fire periodically.
#     """

#     def __init__(self, config: dict, video_dir: str = "videos", verbose: int = 0):
#         super().__init__(verbose)
#         self.config    = config
#         self.video_dir = video_dir
#         os.makedirs(video_dir, exist_ok=True)

#     def _on_step(self) -> bool:
#         # Triggered by EveryNTimesteps — record one rollout
#         env = HMRIEnv(render_mode="rgb_array")
#         env.config.update(self.config)

#         obs, _ = env.reset()
#         frames = []
#         done   = False

#         while not done:
#             frame = env.render()
#             if frame is not None:
#                 frames.append(frame)

#             action, _ = self.model.predict(obs, deterministic=True)
#             obs, _, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated

#         env.close()

#         if frames:
#             path = os.path.join(
#                 self.video_dir,
#                 f"rollout_{self.num_timesteps:08d}.mp4"
#             )
#             imageio.mimsave(path, frames, fps=15)
#             if self.verbose:
#                 print(f"[Video] Saved {path}")

#         return True


# # ─────────────────────────────────────────────────────────────────────────────
# # RANDOM AGENT
# # ─────────────────────────────────────────────────────────────────────────────

# def run_random_agent(config: dict, timesteps: int, log_dir: str):
#     """
#     Rolls out a random agent, logging the same metrics as the trained agents
#     so the midterm overlay plot has a consistent baseline.
#     Returns (mean_return, std_return) over all completed episodes.
#     """
#     logger = configure(log_dir, ["tensorboard", "stdout"])

#     env = HMRIEnv(render_mode=None)
#     env.config.update(config)

#     episode_returns   = []
#     episode_lengths   = []
#     segment_ranks     = []
#     crash_flags       = []

#     obs, _     = env.reset()
#     ep_reward  = 0.0
#     ep_length  = 0
#     ep_seg_rank = 0
#     step       = 0

#     while step < timesteps:
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated

#         ep_reward   += reward
#         ep_length   += 1
#         seg          = info.get("segment", "highway")
#         rank         = SEGMENT_RANK.get(seg, 0)
#         if rank > ep_seg_rank:
#             ep_seg_rank = rank

#         step += 1

#         if done:
#             crashed = bool(info.get("crashed", False))

#             episode_returns.append(ep_reward)
#             episode_lengths.append(ep_length)
#             segment_ranks.append(ep_seg_rank)
#             crash_flags.append(int(crashed))

#             logger.record("random/mean_return",      ep_reward)
#             logger.record("random/episode_length",   ep_length)
#             logger.record("random/segment_reached",  ep_seg_rank)
#             logger.record("random/crashed",          int(crashed))
#             logger.dump(step)

#             obs, _     = env.reset()
#             ep_reward  = 0.0
#             ep_length  = 0
#             ep_seg_rank = 0

#     env.close()

#     mean_r = float(np.mean(episode_returns)) if episode_returns else 0.0
#     std_r  = float(np.std(episode_returns))  if episode_returns else 0.0
#     print(f"\n[Random Agent] Episodes: {len(episode_returns)}")
#     print(f"  Mean return:      {mean_r:.3f} ± {std_r:.3f}")
#     print(f"  Mean seg reached: {np.mean(segment_ranks):.2f}")
#     print(f"  Crash rate:       {np.mean(crash_flags):.2%}")
#     return mean_r, std_r


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────────────────────

# def main(args):

#     base_config = {
#         "duration":             args.ep_length,
#         "simulation_frequency": 5,
#         "policy_frequency":     1,
#         "vehicles_count":       0,       # start with 0 — add NPCs when ready
#         "screen_width":         args.screen_width,
#         "screen_height":        args.screen_height,
#         "offscreen_rendering":  True,
#         "render_agent":         True,
#         "show_trajectories":    False,
#     }

#     tb_log_dir    = f"./tensorboard/hmri_{args.algo}/"
#     video_dir     = f"./videos/{args.algo}/"
#     model_path    = f"{args.algo}_hmri_model"

#     # ── Random agent ─────────────────────────────────────────────────────────
#     if args.algo == "random":
#         run_random_agent(base_config, args.timesteps, f"./tensorboard/hmri_random/")
#         return

#     # ── Vectorised training env ───────────────────────────────────────────────
#     env = DummyVecEnv([make_env_fn(base_config)])

#     # ── Model ─────────────────────────────────────────────────────────────────
#     shared_kwargs = dict(
#         env              = env,
#         verbose          = 1,
#         tensorboard_log  = tb_log_dir,
#         device           = args.device,
#     )

#     if args.algo == "ppo":
#         model = PPO(
#             "MlpPolicy",
#             learning_rate = args.lr,
#             n_steps       = args.n_steps,
#             batch_size    = args.batch_size,
#             # PPO-specific tuning hints for sparse-reward envs:
#             # n_epochs=10, ent_coef=0.01 encourages exploration
#             n_epochs      = 10,
#             ent_coef      = 0.01,
#             **shared_kwargs,
#         )

#     elif args.algo == "dqn":
#         model = DQN(
#             "MlpPolicy",
#             learning_rate          = args.lr,
#             batch_size             = args.batch_size,
#             # Larger replay buffer helps with sparse rewards in long episodes
#             buffer_size            = 50_000,
#             learning_starts        = 1_000,
#             target_update_interval = 500,
#             exploration_fraction   = 0.3,
#             exploration_final_eps  = 0.05,
#             **shared_kwargs,
#         )

#     # ── Callbacks ─────────────────────────────────────────────────────────────
#     metrics_cb = HMRIMetricsCallback(verbose=1)

#     video_cb = EveryNTimesteps(
#         n_steps  = args.video_interval,
#         callback = VideoRecorderCallback(base_config, video_dir=video_dir, verbose=1),
#     )

#     # ── Train ─────────────────────────────────────────────────────────────────
#     model.learn(
#         total_timesteps = args.timesteps,
#         callback        = [metrics_cb, video_cb],
#         tb_log_name     = args.algo,
#     )

#     model.save(model_path)
#     print(f"\n[Training] Model saved to {model_path}.zip")

#     # ── Final video ───────────────────────────────────────────────────────────
#     print("[Training] Recording final rollout...")
#     final_cb = VideoRecorderCallback(base_config, video_dir=video_dir, verbose=1)
#     final_cb.num_timesteps = args.timesteps
#     final_cb.model = model
#     final_cb._on_step()


# # ─────────────────────────────────────────────────────────────────────────────
# # ARGUMENT PARSER
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train RL on stitched HMRI env")

#     parser.add_argument("--algo",           choices=["ppo", "dqn", "random"], default="ppo")
#     parser.add_argument("--timesteps",      type=int,   default=150_000)
#     parser.add_argument("--ep_length",      type=int,   default=200,
#                         help="Max steps per episode (duration in env config)")
#     parser.add_argument("--n_steps",        type=int,   default=1024,
#                         help="PPO: steps collected per update")
#     parser.add_argument("--batch_size",     type=int,   default=64)
#     parser.add_argument("--lr",             type=float, default=3e-4)
#     parser.add_argument("--device",         type=str,   default="auto")
#     parser.add_argument("--screen_width",   type=int,   default=608)
#     parser.add_argument("--screen_height",  type=int,   default=608)
#     parser.add_argument("--video_interval", type=int,   default=50_000,
#                         help="Record a video every N timesteps")

#     args = parser.parse_args()
#     main(args)