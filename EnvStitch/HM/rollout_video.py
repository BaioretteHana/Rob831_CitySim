import argparse
import imageio
import numpy as np

from stable_baselines3 import PPO, DQN

from highway_env.envs.highway_stitchable_env import HighwayEnv
from highway_env.envs.merge_adapter_env import MergeAdapterEnv


# -------------------------------------------------
# Same stitched env (with rendering enabled)
# -------------------------------------------------
class HighwayMergeEnv:

    def __init__(self, config):

        self.highway = HighwayEnv(config=config)
        self.merge = MergeAdapterEnv(config=config)

        self.highway.render_mode = "rgb_array"
        self.merge.render_mode = "rgb_array"

        self.phase = "highway"

    def reset(self):
        self.phase = "highway"
        obs,_ = self.highway.reset()
        return obs

    def step(self, action):

        if self.phase == "highway":
            obs, reward, done, trunc, info = self.highway.step(action)

            if done or trunc:

                if self.highway.vehicle.crashed:
                    return obs, reward, True, trunc, info

                ego = self.highway.vehicle
                handoff = {
                    "speed": ego.speed,
                    "lane": ego.lane_index[2],
                    "position": ego.position[0],
                }

                self.merge.set_handoff_state(handoff)
                obs,_ = self.merge.reset()

                self.phase = "merge"
                return obs, reward, False, False, info

            return obs, reward, False, False, info

        else:
            return self.merge.step(action)

    def render(self):
        if self.phase == "highway":
            return self.highway.render()
        else:
            return self.merge.render()


# -------------------------------------------------
def main(args):

    env = HighwayMergeEnv({"duration": args.ep_length})

    # load model
    if args.algo == "ppo":
        model = PPO.load("ppo_model")
    elif args.algo == "dqn":
        model = DQN.load("dqn_model")
    else:
        model = None

    frames = []

    obs = env.reset()
    done = False

    while not done:

        if model is None:
            action = np.random.randint(5)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, trunc, info = env.step(action)

        frame = env.render()
        frames.append(frame)

    imageio.mimsave(args.output, frames, fps=15)

    print(f"Saved video: {args.output}")


# -------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", choices=["ppo","dqn","random"], default="ppo")
    parser.add_argument("--output", type=str, default="rollout.mp4")
    parser.add_argument("--ep_length", type=int, default=60)

    args = parser.parse_args()

    main(args)