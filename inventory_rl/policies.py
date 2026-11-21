"""Baseline policies, an episode runner and the baseline tuning (all on TUNE_SEEDS)."""

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


def weekday_s_S_policy(s: list[int], S: list[int]) -> Policy:
    """(s, S) with a separate pair for each day of the week (indexed by the day the order is placed)."""

    def act(env: InventoryEnv, obs: np.ndarray) -> int:
        d, ip = env.day, env.inventory_position()
        return int(min(S[d] - ip, env.cfg.max_order)) if ip <= s[d] else 0

    return act


def capped_base_stock_policy(S: int, cap: int) -> Policy:
    """Order up to S (on inventory position), but never more than `cap` units a day (Xin, 2021)."""

    def act(env: InventoryEnv, obs: np.ndarray) -> int:
        return int(np.clip(S - env.inventory_position(), 0, min(cap, env.cfg.max_order)))

    return act


def run_episode(env: InventoryEnv, policy: Policy, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, _ = env.step(policy(env, obs))
        total += r
        done = term or trunc
    return total


def trace_episode(env: InventoryEnv, policy: Policy, seed: int) -> list[dict]:
    """Run one episode and return a per-day log (used by the dashboard)."""
    obs, _ = env.reset(seed=seed)
    rows, done = [], False
    while not done:
        q = policy(env, obs)
        stock_before, weekday = env.stock, env.day
        obs, r, term, trunc, info = env.step(q)
        rows.append({"day": env.t, "weekday": weekday, "stock_start": stock_before, "order": q,
                     **info, "stock_end": env.stock, "cost": -r})
        done = term or trunc
    return rows


def ppo_policy(config: str, seed: int = 0, models_dir: Path = Path("results/models")) -> Policy:
    from stable_baselines3 import PPO  # imported lazily: the env itself does not need torch

    model = PPO.load(models_dir / f"ppo_{config}_seed{seed}", device="cpu")
    return lambda env, obs: int(model.predict(obs, deterministic=True)[0])


def evaluate(config: str, policy: Policy, seeds: list[int]) -> np.ndarray:
    env = InventoryEnv(config)
    return np.array([run_episode(env, policy, s) for s in seeds])


def grid_search_s_S(config: str, seeds: list[int] = TUNE_SEEDS) -> dict:
    """Exhaustive search over 0 <= s < S <= 45."""
    best = {"s": 0, "S": 1, "tune_mean_return": -np.inf}
    for s in range(0, 31):
        for S in range(s + 1, 46):
            mean = evaluate(config, s_S_policy(s, S), seeds).mean()
            if mean > best["tune_mean_return"]:
                best = {"s": s, "S": S, "tune_mean_return": float(mean)}
    return best


def tune_capped_base_stock(config: str, seeds: list[int] = TUNE_SEEDS) -> dict:
    """Exhaustive search over target 0 <= S <= 45 and daily cap 1 <= cap <= max_order."""
    best = {"S": 0, "cap": 1, "tune_mean_return": -np.inf}
    for S in range(0, 46):
        for cap in range(1, InventoryEnv(config).cfg.max_order + 1):
            mean = evaluate(config, capped_base_stock_policy(S, cap), seeds).mean()
            if mean > best["tune_mean_return"]:
                best = {"S": S, "cap": cap, "tune_mean_return": float(mean)}
    return best


def tune_weekday_s_S(config: str, start: dict, seeds: list[int] = TUNE_SEEDS, sweeps: int = 3) -> dict:
    """Coordinate descent: starting from the stationary (s, S), re-tune one weekday's pair at a time
    (s within +/-5 of its current value, S - s in 1..20) holding the other six fixed."""
    s, S = [start["s"]] * 7, [start["S"]] * 7
    best = evaluate(config, weekday_s_S_policy(s, S), seeds).mean()
    for _ in range(sweeps):
        improved = False
        for d in range(7):
            for sd in range(max(0, s[d] - 5), s[d] + 6):
                for gap in range(1, 21):
                    trial_s, trial_S = s.copy(), S.copy()
                    trial_s[d], trial_S[d] = sd, sd + gap
                    mean = evaluate(config, weekday_s_S_policy(trial_s, trial_S), seeds).mean()
                    if mean > best + 1e-9:
                        best, s, S, improved = mean, trial_s, trial_S, True
        if not improved:
            break
    return {"s": s, "S": S, "tune_mean_return": float(best)}


def load_or_tune_baselines(config: str, cache: Path = Path("results/baseline_params.json")) -> dict:
    """Tuned parameters for (s, S), weekday (s, S) and capped base-stock, cached on disk."""
    params = json.loads(cache.read_text()) if cache.exists() else {}
    if config not in params:
        s_S = grid_search_s_S(config)
        params[config] = {"s_S": s_S, "weekday_s_S": tune_weekday_s_S(config, s_S),
                          "capped_base_stock": tune_capped_base_stock(config)}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(params, indent=2))
    return params[config]


def baseline_policies(config: str, cache: Path = Path("results/baseline_params.json")) -> dict[str, Policy]:
    p = load_or_tune_baselines(config, cache)
    return {"(s, S)": s_S_policy(p["s_S"]["s"], p["s_S"]["S"]),
            "weekday (s, S)": weekday_s_S_policy(p["weekday_s_S"]["s"], p["weekday_s_S"]["S"]),
            "capped base-stock": capped_base_stock_policy(p["capped_base_stock"]["S"], p["capped_base_stock"]["cap"])}
