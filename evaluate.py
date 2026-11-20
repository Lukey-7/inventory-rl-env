"""Compare random, (s, S) and PPO on held-out seeds; write results/ table, JSON and plot.

    python evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inventory_rl.env import CONFIGS
from inventory_rl.policies import EVAL_SEEDS, evaluate, load_or_tune_s_S, ppo_policy, random_policy, s_S_policy

RESULTS = Path("results")


def ci95(x: np.ndarray) -> float:
    """Half-width of a normal-approximation 95% confidence interval for the mean."""
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def trained_seeds(cfg: str) -> list[int]:
    return sorted(int(p.stem.rsplit("seed", 1)[1]) for p in (RESULTS / "models").glob(f"ppo_{cfg}_seed*.zip"))


def main() -> None:
    fmt = lambda x: f"{x.mean():.1f} ± {ci95(x):.1f}"
    lines = [f"Mean episode return over {len(EVAL_SEEDS)} held-out seeds (higher is better; "
             "return = minus total cost over 100 days), ± 95% CI. PPO is averaged over its training seeds.", "",
             "| Config | Random | (s, S) tuned | PPO | PPO per training seed | PPO − (s, S), paired |",
             "|---|---|---|---|---|---|"]
    summary = {}
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(13, 3.8))
    for ax, cfg in zip(axes, CONFIGS):
        s, S = load_or_tune_s_S(cfg)
        seeds = trained_seeds(cfg)
        rand = evaluate(cfg, random_policy(0), EVAL_SEEDS)
        ss = evaluate(cfg, s_S_policy(s, S), EVAL_SEEDS)
        per_seed = np.array([evaluate(cfg, ppo_policy(cfg, k), EVAL_SEEDS) for k in seeds])  # (n_agents, n_eval)
        ppo = per_seed.mean(axis=0)  # each eval seed's return averaged over the trained agents
        diff = ppo - ss  # same eval seed => same demand path, so a paired difference is fair
        each = ", ".join(f"{m:.0f}" for m in per_seed.mean(axis=1))
        lines.append(f"| {cfg} | {fmt(rand)} | {fmt(ss)} (s={s}, S={S}) | {fmt(ppo)} | {each} | {fmt(diff)} |")
        summary[cfg] = {"random": rand.mean(), "s_S": ss.mean(), "s": s, "S": S, "ppo": ppo.mean(),
                        "ppo_per_seed": per_seed.mean(axis=1).tolist(), "ppo_minus_s_S": diff.mean(),
                        "ppo_minus_s_S_ci95": ci95(diff)}

        curves = [np.loadtxt(RESULTS / "curves" / f"{cfg}_seed{k}.csv", delimiter=",", skiprows=1) for k in seeds]
        n = min(len(c) for c in curves)
        x, ys = curves[0][:n, 0], np.array([c[:n, 1] for c in curves])
        ax.plot(x, ys.mean(axis=0), label=f"PPO, mean of {len(seeds)} seeds")
        ax.fill_between(x, ys.min(axis=0), ys.max(axis=0), alpha=0.25, label="PPO, min–max over seeds")
        ax.axhline(ss.mean(), color="tab:green", ls="--", label=f"(s, S) = ({s}, {S})")
        ax.set(title=f"{cfg} (L={CONFIGS[cfg].lead_time})", xlabel="timesteps", ylabel="episode return")
        # Early training is far off the chart (stock piles up); zoom to where the comparison is.
        ax.set_ylim(bottom=2 * ss.mean(), top=0.9 * max(ys.max(), ss.mean()))
        ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "learning_curves.png", dpi=120)
    (RESULTS / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "results.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console defaults to cp1252
    print("\n".join(lines))


if __name__ == "__main__":
    main()
