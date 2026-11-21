"""Exact finite-horizon dynamic programming for small configs (the true optimum to measure gaps against).

State: (on-hand stock, pipeline p_1..p_L, day of week), solved backwards over the 100 days, so the
policy also depends on days left. Two approximations, both negligible here:
  * demand is truncated at `max_demand`, with the tail probability lumped into the last value;
  * stock is truncated at `max_stock` (60 units, 15 days of average demand; no good policy gets close).
The state space has (max_stock + 1) * (max_order + 1)^L * 7 states, so this is practical for L <= 2
(easy and medium) but not for the hard config (L = 4: ~83 million states per day).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma

import numpy as np

from inventory_rl.env import InventoryConfig, InventoryEnv


@dataclass
class DPSolution:
    cfg: InventoryConfig
    policy: np.ndarray  # (horizon, stock, p_1, ..., p_L, day) -> optimal order, uint8
    value0: np.ndarray  # optimal expected cost-to-go at t = 0, shape (stock, p_1..p_L, day)
    max_stock: int

    def act(self, env: InventoryEnv, obs: np.ndarray) -> int:
        idx = (env.t, min(env.stock, self.max_stock), *env.pipeline.tolist(), env.day)
        return int(self.policy[idx])

    def expected_return(self) -> float:
        """Exact optimal expected return under the env's random start (stock uniform, empty pipeline)."""
        c = self.cfg
        high = int(2 * c.demand_mean * (c.lead_time + 1))
        start = self.value0[(slice(0, high + 1),) + (0,) * c.lead_time]  # (stock, day)
        return -float(start.mean())


def solve(cfg: InventoryConfig, max_stock: int = 60, max_demand: int = 30) -> DPSolution:
    L, A, X, H = cfg.lead_time, cfg.max_order + 1, max_stock, cfg.horizon
    env = InventoryEnv(cfg)
    rates = np.array([env.demand_rate(d) for d in range(7)])
    D = np.arange(max_demand + 1)
    log_fact = np.array([lgamma(k + 1) for k in D])
    P = np.exp(D[None, :] * np.log(rates[:, None]) - rates[:, None] - log_fact[None, :])  # Poisson pmf, (day, demand)
    P[:, -1] += 1.0 - P.sum(axis=1)  # lump the tail into the last value

    Y = np.arange(X + A)  # stock after the day's delivery arrives, before demand
    left = np.maximum(Y[:, None] - D[None, :], 0)  # (y, demand)
    short = np.maximum(D[None, :] - Y[:, None], 0)
    C = cfg.holding_cost * left @ P.T + cfg.stockout_penalty * short @ P.T  # expected cost, (y, day)
    order_cost = cfg.fixed_cost * (np.arange(A) > 0) + cfg.unit_cost * np.arange(A)
    nxt = np.minimum(left, X)  # next-day stock index

    Yg = np.arange(X + 1)[:, None] + np.arange(A)[None, :]  # (stock, a or p_1) -> y
    V = np.zeros((X + 1,) + (A,) * L + (7,))
    policy = np.zeros((H,) + V.shape, dtype=np.uint8)
    for t in range(H - 1, -1, -1):
        Vn = np.roll(V, -1, axis=-1)  # Vn[..., d] = V[..., d + 1]: tomorrow's weekday
        W = np.einsum("yk...d,dk->y...d", Vn[nxt], P)  # expected cost-to-go given y, (y, next pipeline, day)
        if L == 0:  # today's order arrives today: y = stock + a
            Q = order_cost[None, :, None] + C[Yg] + W[Yg]  # (stock, a, day)
            a_axis = 1
        else:  # p_1 arrives today: y = stock + p_1; the new order joins the end of the pipeline
            Q = W[Yg] + C[Yg].reshape((X + 1, A) + (1,) * L + (7,))  # (stock, p_1, p_2..p_L, a, day)
            Q += order_cost.reshape((A, 1))
            a_axis = -2
        policy[t] = Q.argmin(axis=a_axis)
        V = Q.min(axis=a_axis)
    return DPSolution(cfg, policy, V, X)
