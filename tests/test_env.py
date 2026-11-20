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


def test_holding_cost_and_capacity():
    env = InventoryEnv("easy", capacity=10)
    env.reset(seed=0, options={"initial_stock": 8})
    env.np_random = FixedDemand([1])
    _, r, *_ = env.step(5)  # 8 + 5 capped at 10, sell 1 -> 9 held; order 5 + 2*5
    assert r == -(9 * 1.0 + 15.0)


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
