"""Interactive dashboard: watch a policy, compare policies on the same days, or run the shop yourself.

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from inventory_rl.env import CONFIGS, InventoryEnv
from inventory_rl.dp import solve
from inventory_rl.policies import baseline_policies, constant_policy, ppo_policy, random_policy, trace_episode

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.set_page_config(page_title="Inventory RL", layout="wide")


@st.cache_resource(show_spinner="Loading the trained PPO agents (the first load can take ~30 s)...")
def get_ppo(config: str, seed: int):
    return ppo_policy(config, seed, RESULTS / "models")


@st.cache_resource(show_spinner="Solving for the optimal policy (up to ~30 s)...")
def get_optimal(config: str):
    return solve(CONFIGS[config]).act


def policies(config: str) -> dict:
    out = {f"PPO agent (seed {k})": (lambda k=k: get_ppo(config, k))
           for k in range(3) if (RESULTS / "models" / f"ppo_{config}_seed{k}.zip").exists()}
    if CONFIGS[config].lead_time <= 2:  # exact dynamic programming is only tractable for short lead times
        out["Optimal policy (dynamic programming)"] = lambda: get_optimal(config)
    for name, pol in baseline_policies(config, RESULTS / "baseline_params.json").items():
        out[f"Baseline: {name}"] = lambda pol=pol: pol
    out["Random orders"] = lambda: random_policy(0)
    out["Never order"] = lambda: constant_policy(0)
    return out


def summarise(df: pd.DataFrame) -> dict:
    demand = df["demand"].sum()
    return {"Total cost": df["cost"].sum(), "Fill rate": df["sales"].sum() / demand if demand else 1.0,
            "Lost sales": int(df["lost"].sum()), "Orders placed": int((df["order"] > 0).sum()),
            "Avg stock": df["stock_end"].mean()}


def show_metrics(m: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("Total cost", f"{m['Total cost']:,.0f}")
    cols[1].metric("Fill rate", f"{m['Fill rate']:.1%}", help="Share of customer demand actually served")
    cols[2].metric("Lost sales", m["Lost sales"])
    cols[3].metric("Orders placed", m["Orders placed"])
    cols[4].metric("Avg stock", f"{m['Avg stock']:.1f}")


@st.cache_data(show_spinner=False)
def run(config: str, name: str, seed: int) -> pd.DataFrame:
    return pd.DataFrame(trace_episode(InventoryEnv(config), policies(config)[name](), seed))


# -- sidebar --------------------------------------------------------------------------------
st.sidebar.title("Inventory control")
config = st.sidebar.selectbox("Difficulty", list(CONFIGS), index=2)
cfg = CONFIGS[config]
seed = st.sidebar.number_input("Scenario (random seed)", 0, 10_000, 1000,
                               help="Same seed = same customers, so policies can be compared fairly")
st.sidebar.markdown(
    f"""**This scenario**
- Lead time: **{cfg.lead_time} day(s)**
- Average demand: **{cfg.demand_mean:g}/day**{' with a **weekly cycle**' if cfg.seasonality else ''}
- Costs: holding **{cfg.holding_cost:g}**/unit/day, lost sale **{cfg.stockout_penalty:g}**,
  each order **{cfg.fixed_cost:g}** + **{cfg.unit_cost:g}**/unit
- Episode: **{cfg.horizon} days**""")

watch, compare, play, results = st.tabs(["Watch a policy", "Head to head", "Run the shop yourself", "Results"])

# -- watch ----------------------------------------------------------------------------------
with watch:
    name = st.selectbox("Policy", list(policies(config)), key=f"policy_{config}")
    df = run(config, name, seed)
    show_metrics(summarise(df))
    st.subheader("Stock and demand")
    st.line_chart(df.set_index("day")[["stock_end", "demand", "lost"]]
                  .rename(columns={"stock_end": "stock at close", "lost": "lost sales"}),
                  color=["#3dbb6e", "#4c9be8", "#e8554c"])  # stock green, demand blue, lost red
    st.subheader("Orders placed and deliveries")
    st.bar_chart(df.set_index("day")[["order", "arrived"]], stack=False)
    st.subheader("Where the money went")
    st.bar_chart(df[["holding_cost", "stockout_cost", "order_cost"]].sum().rename("total"), horizontal=True)
    with st.expander("Day-by-day log"):
        st.dataframe(df.assign(weekday=df["weekday"].map(WEEKDAYS.__getitem__)), hide_index=True)

# -- head to head ---------------------------------------------------------------------------
with compare:
    st.write("Every policy faces exactly the same customers (same seed). Lower cost is better.")
    runs = {n: run(config, n, seed) for n in policies(config)}
    table = pd.DataFrame({n: summarise(d) for n, d in runs.items()}).T.sort_values("Total cost")
    st.dataframe(table.style.format({"Total cost": "{:,.0f}", "Fill rate": "{:.1%}", "Avg stock": "{:.1f}"}))
    st.subheader("Cumulative cost over the episode")
    st.line_chart(pd.DataFrame({n: d["cost"].cumsum().values for n, d in runs.items()}, index=runs[name]["day"]))

# -- play -----------------------------------------------------------------------------------
with play:
    key = (config, seed)
    if st.session_state.get("game_key") != key or st.button("Restart"):
        env = InventoryEnv(config)
        env.reset(seed=seed)
        st.session_state.update(game_key=key, env=env, log=[])
    env, log = st.session_state.env, st.session_state.log
    done = env.t >= cfg.horizon

    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"### Day {env.t + 1 if not done else cfg.horizon} of {cfg.horizon} — {WEEKDAYS[env.day]}")
        st.metric("On the shelf", env.stock)
        if cfg.lead_time:
            st.write("On the way: " + ", ".join(f"**{n}** in {i + 1}d" for i, n in enumerate(env.pipeline)))
        st.metric("Cost so far", f"{sum(r['cost'] for r in log):,.0f}")
        if log:
            last = log[-1]
            st.caption(f"Yesterday: {last['demand']} customers, sold {last['sales']}, lost {last['lost']}.")
        if not done:
            with st.form("order"):
                q = st.slider("How many units to order today?", 0, cfg.max_order, 0)
                if st.form_submit_button("Place order and open the shop"):
                    stock_before, weekday = env.stock, env.day
                    _, r, _, _, info = env.step(q)
                    log.append({"day": env.t, "weekday": weekday, "stock_start": stock_before, "order": q,
                                **info, "stock_end": env.stock, "cost": -r})
                    st.rerun()
    with right:
        if log:
            st.line_chart(pd.DataFrame(log).set_index("day")[["stock_end", "demand", "order"]]
                          .rename(columns={"stock_end": "stock at close"}))
    if done:
        mine = summarise(pd.DataFrame(log))
        others = {n: summarise(run(config, n, seed)) for n in policies(config) if not n.startswith("Random")}
        st.success(f"Finished! Your total cost: {mine['Total cost']:,.0f} (fill rate {mine['Fill rate']:.1%}).")
        st.dataframe(pd.DataFrame({"You": mine, **others}).T.sort_values("Total cost")
                     .style.format({"Total cost": "{:,.0f}", "Fill rate": "{:.1%}", "Avg stock": "{:.1f}"}))

# -- results --------------------------------------------------------------------------------
with results:
    if (RESULTS / "results.md").exists():
        st.markdown((RESULTS / "results.md").read_text(encoding="utf-8"))
        st.image(str(RESULTS / "learning_curves.png"))
    else:
        st.info("Run `python train.py` and `python evaluate.py` first.")
