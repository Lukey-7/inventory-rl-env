import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import inventory_rl  # noqa: F401  (registers InventoryControl-v0)
from inventory_rl.env import CONFIGS, InventoryEnv
from inventory_rl.policies import EVAL_SEEDS, constant_policy, evaluate, s_S_policy


class FixedDemand:
    """Stand-in for env.np_random that returns a scripted demand sequence."""

    def __init__(self, demands):
        self.demands = iter(demands)

    def poisson(self, lam):
        return next(self.demands)


@pytest.mark.parametrize("config", list(CONFIGS))
def test_env_checker(config):
    check_env(gym.make("InventoryControl-v0", config=config).unwrapped, skip_render_check=True)


def rollout(seed, actions, config="hard"):
    env = InventoryEnv(config)
    obs, _ = env.reset(seed=seed)
    out = [obs]
    for a in actions:
        obs, r, *_ = env.step(a)
        out += [obs, np.array([r])]
    return np.concatenate(out)


def test_observation_includes_days_left():
    env = InventoryEnv("medium")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.cfg.lead_time + 3,) and obs[-1] == 1.0
    for _ in range(env.cfg.horizon - 1):
        obs, *_ = env.step(0)
    assert obs[-1] == pytest.approx(1 / env.cfg.horizon)
    obs, _, _, truncated, _ = env.step(0)
    assert truncated and obs[-1] == 0.0


def test_determinism_under_seed():
    actions = np.random.default_rng(0).integers(0, 21, size=100)
    assert np.array_equal(rollout(7, actions), rollout(7, actions))
    assert not np.array_equal(rollout(7, actions), rollout(8, actions))


def test_cost_accounting_hand_computed():
    # L=1, h=1, p=10, K=5, c=2; start with 3 units, demands scripted as 4, 7, 1.
    env = InventoryEnv("medium", lead_time=1)
    env.reset(seed=0, options={"initial_stock": 3, "day": 0})
    env.np_random = FixedDemand([4, 7, 1])
    # Day 1: order 2 (5 + 2*2 = 9); nothing arrives; sell 3, lose 1 (10); stock 0.  -> 19
    _, r1, _, _, i1 = env.step(2)
    # Day 2: order 0; the 2 units arrive; sell 2, lose 5 (50); stock 0.             -> 50
    _, r2, _, _, i2 = env.step(0)
    # Day 3: order 5 (5 + 2*5 = 15); nothing arrives; lose 1 (10); stock 0.         -> 25
    _, r3, _, _, i3 = env.step(5)
    assert (r1, r2, r3) == (-19.0, -50.0, -25.0)
    assert (i1["lost"], i2["lost"], i3["lost"]) == (1, 5, 1)
    assert i2["arrived"] == 2 and env.stock == 0 and env.pipeline.tolist() == [5]


def test_holding_cost_and_no_silent_overflow():
    env = InventoryEnv("easy", stock_scale=10)
    env.reset(seed=0, options={"initial_stock": 8})
    env.np_random = FixedDemand([1])
    obs, r, *_ = env.step(5)  # 8 + 5 arrive, sell 1 -> 12 held (none discarded); order 5 + 2*5
    assert env.stock == 12 and r == -(12 * 1.0 + 15.0)
    assert obs[0] == 1.0  # the observation saturates instead


@pytest.mark.parametrize("lead_time", [0, 1, 2, 4])
def test_pipeline_arrives_after_exactly_L_steps(lead_time):
    env = InventoryEnv("medium", lead_time=lead_time, demand_mean=0.0)
    env.reset(seed=0, options={"initial_stock": 0})
    arrived = [env.step(3 if t == 0 else 0)[4]["arrived"] for t in range(lead_time + 3)]
    expected = [0] * (lead_time + 3)
    expected[lead_time] = 3
    assert arrived == expected
    assert env.stock == 3


@pytest.mark.parametrize("config", list(CONFIGS))
def test_degenerate_policies_score_badly(config):
    L = CONFIGS[config].lead_time
    s = int(4 * (L + 1))  # hand-picked, untuned (s, S): reorder at the mean lead-time demand
    sensible = evaluate(config, s_S_policy(s, s + 10), EVAL_SEEDS[:10]).mean()
    never = evaluate(config, constant_policy(0), EVAL_SEEDS[:10]).mean()
    flood = evaluate(config, constant_policy(CONFIGS[config].max_order), EVAL_SEEDS[:10]).mean()
    # Returns are negative, so 1.4x the return means at least 40% more cost than (s, S).
    assert never < 1.4 * sensible and flood < 1.4 * sensible


def test_dp_optimum_matches_simulation_and_beats_baselines():
    from inventory_rl.dp import solve
    from inventory_rl.env import InventoryConfig
    from inventory_rl.policies import run_episode

    cfg = InventoryConfig(lead_time=1, demand_mean=1.5, max_order=6, horizon=8, seasonality=0.5)
    sol = solve(cfg, max_stock=20, max_demand=15)
    env, seeds = InventoryEnv(cfg), range(2000)
    dp = np.array([run_episode(env, sol.act, s) for s in seeds])
    stderr = dp.std(ddof=1) / np.sqrt(len(dp))
    assert abs(dp.mean() - sol.expected_return()) < 4 * stderr  # exact value agrees with Monte Carlo
    for s, S in [(0, 3), (1, 4), (2, 6), (3, 8)]:  # and (s, S) policies do no better on the same seeds
        assert np.mean([run_episode(env, s_S_policy(s, S), k) for k in seeds]) <= dp.mean()
