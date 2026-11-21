"""Compare PPO with tuned baselines and (where tractable) the exact optimum on held-out seeds.

    python evaluate.py
Writes results/results.md, results/results.json and results/learning_curves.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inventory_rl.dp import solve
from inventory_rl.env import CONFIGS
from inventory_rl.policies import EVAL_SEEDS, baseline_policies, evaluate, load_or_tune_baselines, ppo_policy, random_policy

RESULTS = Path("results")
DP_MAX_LEAD_TIME = 2  # hard (L = 4) has ~83M states per day: not tractable here
BASELINE_KEYS = {"(s, S)": "s_S", "weekday (s, S)": "weekday_s_S", "capped base-stock": "capped_base_stock"}


def ci95(x: np.ndarray) -> float:
    """Half-width of a normal-approximation 95% confidence interval for the mean."""
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def trained_seeds(cfg: str) -> list[int]:
    return sorted(int(p.stem.rsplit("seed", 1)[1]) for p in (RESULTS / "models").glob(f"ppo_{cfg}_seed*.zip"))


def main() -> None:
    fmt = lambda x: f"{x.mean():.1f} ± {ci95(x):.1f}"
    table = ["| Config | Optimal (DP) | (s, S) | weekday (s, S) | capped base-stock | PPO (3 seeds) "
             "| PPO per seed | PPO − strongest baseline |", "|---|---|---|---|---|---|---|---|"]
    gaps = ["| Config | (s, S) | weekday (s, S) | capped base-stock | PPO (3 seeds) |", "|---|---|---|---|---|"]
    random_note, summary = [], {}
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(13, 3.8))
    for ax, cfg in zip(axes, CONFIGS):
        params = load_or_tune_baselines(cfg)
        base = {name: evaluate(cfg, pol, EVAL_SEEDS) for name, pol in baseline_policies(cfg).items()}
        # "Strongest" is chosen on the tuning seeds, never on the evaluation seeds.
        strongest = max(base, key=lambda n: params[BASELINE_KEYS[n]]["tune_mean_return"])
        per_seed = np.array([evaluate(cfg, ppo_policy(cfg, k), EVAL_SEEDS) for k in trained_seeds(cfg)])
        ppo = per_seed.mean(axis=0)  # each eval seed's return averaged over the trained agents
        dp = evaluate(cfg, solve(CONFIGS[cfg]).act, EVAL_SEEDS) if CONFIGS[cfg].lead_time <= DP_MAX_LEAD_TIME else None
        rand = evaluate(cfg, random_policy(0), EVAL_SEEDS)

        cells = [fmt(dp) if dp is not None else "not tractable"]
        cells += [fmt(v) + (" ★" if n == strongest else "") for n, v in base.items()]
        cells += [fmt(ppo), ", ".join(f"{m:.0f}" for m in per_seed.mean(axis=1)), f"{fmt(ppo - base[strongest])}"]
        table.append(f"| {cfg} | " + " | ".join(cells) + " |")
        if dp is not None:  # extra cost relative to the optimum, paired on the same seeds
            pct = lambda x: f"{100 * (dp - x).mean() / -dp.mean():.1f}% ± {100 * ci95(dp - x) / -dp.mean():.1f}"
            gaps.append(f"| {cfg} | " + " | ".join(pct(v) for v in [*base.values(), ppo]) + " |")
        random_note.append(f"{cfg} {rand.mean():,.0f}")
        summary[cfg] = {"optimal_dp": None if dp is None else dp.mean(), "ppo": ppo.mean(),
                        "ppo_per_seed": per_seed.mean(axis=1).tolist(), "strongest_baseline": strongest,
                        "ppo_minus_strongest": (ppo - base[strongest]).mean(),
                        "ppo_minus_strongest_ci95": ci95(ppo - base[strongest]), "random": rand.mean(),
                        **{n: v.mean() for n, v in base.items()}, "params": params}

        curves = [np.loadtxt(RESULTS / "curves" / f"{cfg}_seed{k}.csv", delimiter=",", skiprows=1)
                  for k in trained_seeds(cfg)]
        n = min(len(c) for c in curves)
        x, ys = curves[0][:n, 0], np.array([c[:n, 1] for c in curves])
        ax.plot(x, ys.mean(axis=0), label=f"PPO, mean of {len(curves)} seeds")
        ax.fill_between(x, ys.min(axis=0), ys.max(axis=0), alpha=0.25, label="PPO, min–max over seeds")
        ax.axhline(base[strongest].mean(), color="tab:green", ls="--", label=f"best baseline: {strongest}")
        if dp is not None:
            ax.axhline(dp.mean(), color="black", ls=":", label="optimal (DP)")
        ax.set(title=f"{cfg} (L={CONFIGS[cfg].lead_time})", xlabel="timesteps", ylabel="episode return")
        # Early training is far off the chart (stock piles up); zoom to where the comparison is.
        ax.set_ylim(bottom=2 * base[strongest].mean(), top=0.9 * max(ys.max(), base[strongest].mean()))
        ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "learning_curves.png", dpi=120)

    lines = [f"Mean episode return over {len(EVAL_SEEDS)} held-out seeds (higher is better; return = minus "
             "total cost over 100 days), ± 95% CI. ★ = strongest baseline, chosen on the tuning seeds. "
             "The last column is paired (same demand on each seed).", "", *table, "",
             "Extra cost compared with the optimal policy (paired, ± 95% CI); lower is better:", "", *gaps, "",
             "Random ordering, for scale: " + ", ".join(random_note) + "."]
    (RESULTS / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "results.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console defaults to cp1252
    print("\n".join(lines))


if __name__ == "__main__":
    main()
