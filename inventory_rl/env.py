"""Single-product, lost-sales inventory control as a Gymnasium environment.

Timeline of one day (one step):
  1. The agent chooses an order quantity q (0..max_order) and pays for it
     (fixed cost K if q > 0, plus unit cost c * q).
  2. The order placed L days ago arrives (with L = 0, today's order arrives now).
     Nothing is ever thrown away: stock has no hard cap.
  3. Poisson demand D is drawn; sales = min(stock, D); the rest is lost.
  4. Holding cost h is charged on the stock left at the end of the day.

reward = -(h * stock_end + p * lost + K * [q > 0] + c * q)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class InventoryConfig:
    lead_time: int = 0
    demand_mean: float = 4.0
    seasonality: float = 0.0  # amplitude of the weekly demand cycle (0 = stationary)
    max_order: int = 20
    stock_scale: int = 50  # only used to scale the observation; stock itself is uncapped
    holding_cost: float = 1.0
    stockout_penalty: float = 10.0
    fixed_cost: float = 5.0
    unit_cost: float = 2.0
    horizon: int = 100


CONFIGS: dict[str, InventoryConfig] = {
    "easy": InventoryConfig(lead_time=0),
    "medium": InventoryConfig(lead_time=2),
    "hard": InventoryConfig(lead_time=4, seasonality=0.5, fixed_cost=20.0),
}


class InventoryEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: str | InventoryConfig = "easy", render_mode: str | None = None, **overrides):
        cfg = CONFIGS[config] if isinstance(config, str) else config
        self.cfg = replace(cfg, **overrides)
        if self.cfg.lead_time < 0 or self.cfg.max_order < 1 or self.cfg.stock_scale < 1:
            raise ValueError(f"invalid config: {self.cfg}")
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(self.cfg.max_order + 1)
        # [min(stock / stock_scale, 1), pipeline_1..L / max_order, day_of_week / 6, days_left / horizon]
        # Days left makes the fixed-horizon problem Markov: what is optimal on day 99 differs from day 9.
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.cfg.lead_time + 3,), dtype=np.float32)
        self.stock = 0
        self.pipeline = np.zeros(self.cfg.lead_time, dtype=np.int64)
        self.day = 0
        self.t = 0

    # -- helpers -----------------------------------------------------------
    def demand_rate(self, day_of_week: int) -> float:
        c = self.cfg
        return c.demand_mean * (1.0 + c.seasonality * np.sin(2 * np.pi * day_of_week / 7))

    def inventory_position(self) -> int:
        """On-hand stock plus everything already ordered but not yet arrived."""
        return int(self.stock + self.pipeline.sum())

    def _obs(self) -> np.ndarray:
        c = self.cfg
        stock = min(self.stock / c.stock_scale, 1.0)  # saturates; no sensible policy holds that much
        days_left = (c.horizon - self.t) / c.horizon
        obs = np.concatenate(([stock], self.pipeline / c.max_order, [self.day / 6, days_left]))
        return obs.astype(np.float32)

    # -- Gymnasium API -----------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        c = self.cfg
        # Random starting stock (up to ~2 lead-time demands) unless pinned by options.
        high = int(2 * c.demand_mean * (c.lead_time + 1))
        self.stock = int(options.get("initial_stock", self.np_random.integers(0, high + 1)))
        self.pipeline = np.zeros(c.lead_time, dtype=np.int64)
        self.day = int(self.np_random.integers(0, 7)) if "day" not in options else int(options["day"])
        self.t = 0
        return self._obs(), {"stock": self.stock}

    def step(self, action):
        c = self.cfg
        q = int(action)
        if not 0 <= q <= c.max_order:
            raise ValueError(f"action {action} outside 0..{c.max_order}")

        order_cost = (c.fixed_cost if q > 0 else 0.0) + c.unit_cost * q
        if c.lead_time == 0:
            arriving = q
        else:
            arriving = int(self.pipeline[0])
            self.pipeline = np.append(self.pipeline[1:], q)
        self.stock += arriving

        demand = int(self.np_random.poisson(self.demand_rate(self.day)))
        sales = min(self.stock, demand)
        lost = demand - sales
        self.stock -= sales

        holding = c.holding_cost * self.stock
        stockout = c.stockout_penalty * lost
        cost = holding + stockout + order_cost
        self.day = (self.day + 1) % 7
        self.t += 1

        info = {"demand": demand, "sales": sales, "lost": lost, "arrived": arriving,
                "holding_cost": holding, "stockout_cost": stockout, "order_cost": order_cost}
        # Fixed-length episodes: the horizon is a time limit, so it is truncation.
        return self._obs(), -float(cost), False, self.t >= c.horizon, info

    def render(self):
        if self.render_mode == "ansi":
            return f"t={self.t} stock={self.stock} pipeline={self.pipeline.tolist()} day={self.day}"
        return None

    def config_dict(self) -> dict:
        return asdict(self.cfg)
