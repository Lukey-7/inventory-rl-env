"""Baseline policies, an episode runner and the (s, S) grid search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from inventory_rl.env import InventoryEnv

Policy = Callable[[InventoryEnv, np.ndarray], int]

TUNE_SEEDS = list(range(100, 120))  # used only for grid search
EVAL_SEEDS = list(range(1000, 1030))  # held out, used only for the final comparison


def random_policy(seed: int = 0) -> Policy:
    rng = np.random.default_rng(seed)
    return lambda env, obs: int(rng.integers(0, env.cfg.max_order + 1))


def constant_policy(q: int) -> Policy:
    return lambda env, obs: q


def s_S_policy(s: int, S: int) -> Policy:
    """Classic (s, S): when the inventory position falls to s or below, order up to S."""

    def act(env: InventoryEnv, obs: np.ndarray) -> int:
        ip = env.inventory_position()
        return int(min(S - ip, env.cfg.max_order)) if ip <= s else 0

    return act


def run_episode(env: InventoryEnv, policy: Policy, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, _ = env.step(policy(env, obs))
        total += r
        done = term or trunc
    return total


def evaluate(config: str, policy: Policy, seeds: list[int]) -> np.ndarray:
    env = InventoryEnv(config)
    return np.array([run_episode(env, policy, s) for s in seeds])


def grid_search_s_S(config: str, seeds: list[int] = TUNE_SEEDS) -> tuple[int, int, float]:
    """Exhaustive search over 0 <= s < S <= 45 on the tuning seeds."""
    best = (0, 1, -np.inf)
    for s in range(0, 31):
        for S in range(s + 1, 46):
            mean = evaluate(config, s_S_policy(s, S), seeds).mean()
            if mean > best[2]:
                best = (s, S, float(mean))
    return best


def load_or_tune_s_S(config: str, cache: Path = Path("results/s_S_params.json")) -> tuple[int, int]:
    params = json.loads(cache.read_text()) if cache.exists() else {}
    if config not in params:
        s, S, mean = grid_search_s_S(config)
        params[config] = {"s": s, "S": S, "tune_mean_return": round(mean, 2)}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(params, indent=2))
    return params[config]["s"], params[config]["S"]
