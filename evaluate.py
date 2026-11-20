"""Compare random, (s, S) and PPO on held-out seeds; write results/ table and plot.

    python evaluate.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from inventory_rl.env import CONFIGS
from inventory_rl.policies import EVAL_SEEDS, evaluate, load_or_tune_s_S, random_policy, s_S_policy

RESULTS = Path("results")


def ci95(x: np.ndarray) -> float:
    """Half-width of a normal-approximation 95% confidence interval for the mean."""
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def ppo_policy(model: PPO):
    return lambda env, obs: int(model.predict(obs, deterministic=True)[0])


def main() -> None:
    lines = [f"Mean episode return over {len(EVAL_SEEDS)} held-out seeds (higher is better; "
             "return = minus total cost over 100 days), +/- 95% CI.", "",
             "| Config | Random | (s, S) tuned | PPO | PPO - (s, S), paired |",
             "|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(13, 3.8))
    for ax, cfg in zip(axes, CONFIGS):
        s, S = load_or_tune_s_S(cfg)
        rand = evaluate(cfg, random_policy(0), EVAL_SEEDS)
        ss = evaluate(cfg, s_S_policy(s, S), EVAL_SEEDS)
        ppo = evaluate(cfg, ppo_policy(PPO.load(RESULTS / f"ppo_{cfg}", device="cpu")), EVAL_SEEDS)
        diff = ppo - ss  # same seeds => same demand paths, so a paired difference is fair
        fmt = lambda x: f"{x.mean():.1f} +/- {ci95(x):.1f}"
        lines.append(f"| {cfg} | {fmt(rand)} | {fmt(ss)} (s={s}, S={S}) | {fmt(ppo)} | {fmt(diff)} |")

        curve = np.loadtxt(RESULTS / f"curve_{cfg}.csv", delimiter=",", skiprows=1)
        ax.plot(curve[:, 0], curve[:, 1], label="PPO (training episodes)")
        ax.axhline(ss.mean(), color="tab:green", ls="--", label=f"(s, S) = ({s}, {S})")
        ax.set(title=f"{cfg} (L={CONFIGS[cfg].lead_time})", xlabel="timesteps", ylabel="episode return")
        ax.set_ylim(bottom=min(curve[:, 1].min(), ss.mean()) * 1.1, top=max(curve[:, 1].max(), ss.mean()) * 0.9)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "learning_curves.png", dpi=120)
    (RESULTS / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
