# inventory-rl-env

A small, self-contained reinforcement-learning project: a [Gymnasium](https://gymnasium.farama.org/)
environment for **single-product inventory control** with lost sales and an order lead time, three
difficulty configs, a random baseline, the classic operations-research **(s, S)** policy (grid-searched
per config) and a **PPO** agent from stable-baselines3, all compared over held-out seeds with
confidence intervals.

## The MDP

Each step is one day. The agent decides how many units to order.

| | |
|---|---|
| **Action** | `Discrete(max_order + 1)`: order quantity `q ∈ {0, …, 20}` |
| **Observation** | `Box([0, 1]^(L+2))`: on-hand stock / capacity, each of the `L` outstanding orders / max_order, day of week / 6 |
| **Dynamics** | Pay for the order; the order placed `L` days ago arrives (with `L = 0` it arrives at once); stock is capped at `capacity = 50`; demand `D ~ Poisson(μ_d)` is drawn; `sales = min(stock, D)`; unmet demand is **lost** |
| **Demand** | `μ_d = μ · (1 + a · sin(2π · d / 7))` with `μ = 4` and weekly amplitude `a` (0 = stationary) |
| **Reward** | `−(h · stock_end + p · lost + K · [q > 0] + c · q)` with `h = 1`, `p = 10`, `c = 2`, `K` per config |
| **Episode** | 100 days, then `truncated = True` (a time limit, never `terminated`). Starting stock is random in `[0, 2μ(L+1)]`, starting weekday random; both come from the seeded RNG |

| Config | Lead time `L` | Seasonality `a` | Fixed cost `K` |
|---|---|---|---|
| easy | 0 | 0 | 5 |
| medium | 2 | 0 | 5 |
| hard | 4 | 0.5 | 20 |

```python
import gymnasium as gym
import inventory_rl  # registers the env

env = gym.make("InventoryControl-v0", config="hard")          # or "easy" / "medium"
env = gym.make("InventoryControl-v0", config="medium", lead_time=3, fixed_cost=10)  # override any field
```

## Usage

Install (Python 3.10–3.12; developed on 3.12 in a `uv` virtual environment):

```bash
pip install -e ".[test]"
```

Train PPO on all three configs (CPU, about 9 minutes in total; 300k steps each):

```bash
python train.py --steps 300000
```

Evaluate random vs (s, S) vs PPO, and write `results/results.md` and `results/learning_curves.png`:

```bash
python evaluate.py
```

The trained models (`results/ppo_*.zip`) and the tuned (s, S) parameters are committed, so
`evaluate.py` runs without retraining. Tests: `pytest`.

## Results

Mean episode return over **30 held-out seeds** (seeds 1000–1029; the (s, S) grid search used
seeds 100–119), ± the half-width of a normal-approximation 95% confidence interval. Returns are minus
the total cost over 100 days, so higher is better. All policies see the same demand paths for a given
seed, so the last column is a **paired** difference (common random numbers).

| Config | Random | (s, S) tuned | PPO | PPO − (s, S), paired |
|---|---|---|---|---|
| easy | −6915.2 ± 54.9 | −1541.5 ± 21.8 (s=3, S=9) | −1546.7 ± 17.9 | −5.2 ± 14.0 |
| medium | −6861.0 ± 53.7 | −1741.0 ± 28.3 (s=12, S=18) | −1724.3 ± 26.6 | +16.7 ± 18.9 |
| hard | −8251.4 ± 67.4 | −2442.4 ± 51.5 (s=17, S=29) | −2315.3 ± 39.6 | **+127.1 ± 36.1** |

![Learning curves](results/learning_curves.png)

**Does PPO beat (s, S)?** Honestly: only on *hard*.

- **easy, medium:** PPO matches the tuned (s, S) policy; the paired difference's confidence interval
  contains zero in both. With stationary demand, (s, S) is close to optimal, so there is little to gain.
- **hard:** PPO is better by about 127 cost units per episode (≈ 5%), and the interval excludes zero.
  The most likely reason is seasonality: a single stationary (s, S) pair cannot react to the day of the
  week, while PPO observes it. This is a weakness of the baseline as much as a strength of PPO — a
  day-of-week-dependent (s_d, S_d) policy would be the fairer comparison and might close the gap.
- These are single training runs (one PPO seed per config). Variation across training seeds was not
  measured.

## Design choices

- **Lost sales rather than backorders.** Backorder models give (s, S) a clean optimality story;
  lost sales with a positive lead time does not have a simple optimal policy, which makes it a more
  interesting test for RL.
- **Normalised, bounded observations.** Everything is scaled into `[0, 1]`, so PPO needs no
  observation normalisation and the env passes `check_env` with a tight `Box`.
- **Rewards in raw cost units.** The environment reports true costs (daily values in the tens), and
  `train.py` rescales them with `VecNormalize(norm_reward=True)` for learning only; evaluation always
  uses the raw environment, so all policies are scored identically.
- **(s, S) on inventory position.** The baseline compares on-hand stock *plus* the pipeline against `s`,
  which is the standard form when there is a lead time. It is tuned by exhaustive grid search
  (`0 ≤ s ≤ 30`, `s < S ≤ 45`) on separate tuning seeds.
- **Fixed horizon as truncation.** Episodes end on a time limit, so PPO bootstraps through the last
  step rather than treating day 100 as the end of the world.
- **Reproducibility.** All randomness comes from the env's seeded `np_random`; the policies never
  touch it, which is what makes the paired comparison valid.

## Tests

`pytest` (13 tests, about a second):

- `check_env` passes on all three configs;
- identical seeds and actions give identical trajectories, and different seeds differ;
- cost accounting on a hand-computed three-day episode (scripted demand, lost sales, an arrival);
- holding cost and the capacity cap;
- an order arrives after exactly `L` steps, for `L ∈ {0, 1, 2, 4}`;
- reward sanity: "never order" and "order the maximum every day" both cost at least 40% more than an
  untuned (s, S) policy.

## Limitations

- One product, one location, no perishability, no supplier uncertainty; demand is Poisson with a known
  form.
- The capacity cap silently discards overflow (it is still paid for), which a real warehouse would not.
- Short training budget (300k steps per config) and a single PPO seed; no hyperparameter search.
- The 95% intervals use a normal approximation with 30 episodes.
- The (s, S) baseline is stationary; a stronger OR baseline for the seasonal case is left out.

## Layout

```
inventory_rl/env.py       environment and the three configs
inventory_rl/policies.py  random, constant and (s, S) policies, episode runner, grid search
train.py                  PPO training (writes results/ppo_*.zip and learning-curve CSVs)
evaluate.py               comparison table and plot
tests/test_env.py         tests
```

## Licence

MIT.
