"""Train PPO agents (one per difficulty config and seed) on CPU.

    python train.py                         # 3 configs x 3 seeds, 150k steps each
    python train.py --configs hard --seeds 0 --steps 50000
    python train.py --jobs 3                # run 3 trainings in parallel processes
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from inventory_rl.env import CONFIGS, InventoryEnv

RESULTS = Path("results")


class CurveLogger(BaseCallback):
    """After each rollout, record the mean raw (unnormalised) return of recent episodes."""

    def __init__(self):
        super().__init__()
        self.rows: list[tuple[int, float]] = []

    def _on_rollout_end(self) -> None:
        if self.model.ep_info_buffer:
            self.rows.append((self.num_timesteps, float(np.mean([e["r"] for e in self.model.ep_info_buffer]))))

    def _on_step(self) -> bool:
        return True


def train(config: str, steps: int, seed: int) -> str:
    torch.set_num_threads(1)  # small MLP: extra threads only add overhead on CPU
    # Monitor (inside make_vec_env) logs raw returns; VecNormalize only rescales the
    # training reward, which matters because daily costs are in the tens.
    venv = make_vec_env(InventoryEnv, n_envs=4, seed=seed, env_kwargs={"config": config})
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=256, n_epochs=10, gamma=0.99,
                gae_lambda=0.95, ent_coef=0.01, learning_rate=3e-4, seed=seed, verbose=0,
                policy_kwargs={"net_arch": [64, 64]}, device="cpu")
    logger = CurveLogger()
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=logger)
    model.save(RESULTS / "models" / f"ppo_{config}_seed{seed}")
    with open(RESULTS / "curves" / f"{config}_seed{seed}.csv", "w", newline="") as f:
        csv.writer(f).writerows([("timesteps", "mean_episode_return"), *logger.rows])
    return (f"{config} seed {seed}: {steps} steps in {time.time() - t0:.0f}s, "
            f"final training return {logger.rows[-1][1]:.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--jobs", type=int, default=1)
    args = p.parse_args()
    for sub in ("models", "curves"):
        (RESULTS / sub).mkdir(parents=True, exist_ok=True)
    runs = list(product(args.configs, [args.steps], args.seeds))
    with ProcessPoolExecutor(args.jobs) as pool:
        for line in pool.map(train, *zip(*runs)):
            print(line, flush=True)
