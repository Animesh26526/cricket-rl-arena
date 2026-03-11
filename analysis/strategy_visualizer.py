"""
analysis/strategy_visualizer.py
---------------------------------
Generates strategy heatmaps and shot-preference tables from a trained agent.

Outputs (saved to analysis/plots/)
-------
  1. shot_delivery_heatmap.png
       Rows = deliveries, Columns = shots, Cells = Q-value of that (delivery, shot) pair.
  2. preferred_shots_bar.png
       Bar chart of the most preferred shot per delivery.
  3. pressure_heatmap.png
       How shot preferences change as required_run_rate increases.

Usage
-----
    python analysis/strategy_visualizer.py                     # best checkpoint
    python analysis/strategy_visualizer.py --model path/to/model.pkl
    python analysis/strategy_visualizer.py --agent dqn
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from environment.cricket_env import CricketEnv
from environment.probability_engine import ALL_DELIVERIES

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "training" / "checkpoints"
PLOT_DIR = Path(__file__).resolve().parent / "plots"
TOTAL_BALLS = {"T20": 120, "ODI": 300, "Test": 450}


# ---------------------------------------------------------------------------
# Helper: run greedy episodes and collect shot data
# ---------------------------------------------------------------------------


def collect_shot_data(
    agent,
    match_format: str,
    n_episodes: int,
) -> Dict:
    """
    Run `n_episodes` greedy innings and collect (delivery, shot, reward) tuples.
    Returns dict keyed by delivery with shot→count and shot→avg_reward.
    """
    total_balls = TOTAL_BALLS.get(match_format, 120)
    env = CricketEnv(total_balls=total_balls, match_format=match_format)
    shot_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    shot_rewards: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            avail = env.get_available_actions()
            action = agent.greedy_action(state, avail)
            shot = (
                env.available_shots[action]
                if action < len(env.available_shots)
                else "Unknown"
            )
            delivery = env.delivery
            state, reward, done, _ = env.step(action)
            shot_counts[delivery][shot] += 1
            shot_rewards[delivery][shot].append(reward)

    return {"counts": dict(shot_counts), "rewards": dict(shot_rewards)}


# ---------------------------------------------------------------------------
# Plot 1: Q-value heatmap  (delivery × shot)
# ---------------------------------------------------------------------------


def plot_shot_delivery_heatmap(data: Dict, save_path: Path) -> None:
    counts = data["counts"]

    # Build matrix: rows=deliveries, cols=all shots seen
    all_shots_seen = sorted({s for d in counts.values() for s in d})
    matrix = np.zeros((len(ALL_DELIVERIES), len(all_shots_seen)))
    for di, delivery in enumerate(ALL_DELIVERIES):
        total = sum(counts.get(delivery, {}).values()) or 1
        for si, shot in enumerate(all_shots_seen):
            matrix[di, si] = counts.get(delivery, {}).get(shot, 0) / total * 100

    fig, ax = plt.subplots(figsize=(max(10, len(all_shots_seen) * 0.85), 7))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=all_shots_seen,
        yticklabels=ALL_DELIVERIES,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="#cccccc",
        cbar_kws={"label": "Shot frequency (%)"},
    )
    ax.set_title(
        "Shot Selection Heatmap — Frequency (%) per Delivery", fontsize=13, pad=12
    )
    ax.set_xlabel("Shot")
    ax.set_ylabel("Delivery")
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualizer] Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 2: Preferred shots bar chart
# ---------------------------------------------------------------------------


def plot_preferred_shots_bar(data: Dict, save_path: Path) -> None:
    counts = data["counts"]
    deliveries = ALL_DELIVERIES
    top_shots = []
    top_pcts = []

    for d in deliveries:
        sc = counts.get(d, {})
        if not sc:
            top_shots.append("N/A")
            top_pcts.append(0)
            continue
        best = max(sc, key=sc.get)
        total = sum(sc.values()) or 1
        top_shots.append(best)
        top_pcts.append(sc[best] / total * 100)

    # Colour by shot type
    palette = plt.cm.get_cmap("tab20", len(set(top_shots)))
    shot_colour = {s: palette(i) for i, s in enumerate(sorted(set(top_shots)))}
    colours = [shot_colour[s] for s in top_shots]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(deliveries, top_pcts, color=colours, edgecolor="white", height=0.65)
    for bar, shot, pct in zip(bars, top_shots, top_pcts):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{shot} ({pct:.0f}%)",
            va="center",
            fontsize=8,
        )

    ax.set_xlim(0, 110)
    ax.set_xlabel("Preferred Shot Frequency (%)")
    ax.set_title("Most Preferred Shot per Delivery (Greedy Agent)", fontsize=13)
    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in shot_colour.items()]
    ax.legend(handles=legend_patches, fontsize=7, loc="lower right", ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualizer] Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 3: Pressure vs shot-aggression heatmap
# ---------------------------------------------------------------------------


def plot_pressure_heatmap(agent, match_format: str, save_path: Path) -> None:
    """
    Show how shot selection changes as RRR increases.
    Simulate innings with increasing target pressures.
    """
    total_balls = TOTAL_BALLS.get(match_format, 120)
    rrr_bands = [
        (0, 6, "RRR 0-6"),
        (6, 9, "RRR 6-9"),
        (9, 12, "RRR 9-12"),
        (12, 36, "RRR 12+"),
    ]
    n_episodes = 300

    # Run episodes with various targets to hit different RRR windows
    targets = [70, 100, 130, 170]
    all_shots_seen = set()
    band_shot_counts: List[Dict] = [{} for _ in rrr_bands]

    for t_idx, target in enumerate(targets):
        env = CricketEnv(
            total_balls=total_balls, match_format=match_format, target=target
        )
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            while not done:
                avail = env.get_available_actions()
                action = agent.greedy_action(state, avail)
                shot = (
                    env.available_shots[action]
                    if action < len(env.available_shots)
                    else "Unknown"
                )

                # Which RRR band are we in?
                rrr = state[6]
                band_idx = min(t_idx, len(rrr_bands) - 1)
                band_shot_counts[band_idx][shot] = (
                    band_shot_counts[band_idx].get(shot, 0) + 1
                )
                all_shots_seen.add(shot)
                state, _, done, _ = env.step(action)

    all_shots_seen = sorted(all_shots_seen)
    matrix = np.zeros((len(rrr_bands), len(all_shots_seen)))
    for bi, bsc in enumerate(band_shot_counts):
        total = sum(bsc.values()) or 1
        for si, shot in enumerate(all_shots_seen):
            matrix[bi, si] = bsc.get(shot, 0) / total * 100

    fig, ax = plt.subplots(figsize=(max(8, len(all_shots_seen) * 0.9), 4))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=all_shots_seen,
        yticklabels=[b[2] for b in rrr_bands],
        annot=True,
        fmt=".0f",
        cmap="Blues",
        linewidths=0.4,
        cbar_kws={"label": "Shot frequency (%)"},
    )
    ax.set_title("Shot Selection vs Chase Pressure (RRR bands)", fontsize=12)
    ax.set_xlabel("Shot")
    ax.set_ylabel("Required Run Rate Band")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualizer] Saved: {save_path}")


# ---------------------------------------------------------------------------
# Print strategy table to console
# ---------------------------------------------------------------------------


def print_strategy_table(data: Dict) -> None:
    counts = data["counts"]
    print("\n  STRATEGY TABLE — Best Shot per Delivery")
    print("  " + "─" * 50)
    print(f"  {'Delivery':<22} {'Best Shot':<22} {'Freq':>6}")
    print("  " + "─" * 50)
    for delivery in ALL_DELIVERIES:
        sc = counts.get(delivery, {})
        if not sc:
            continue
        best = max(sc, key=sc.get)
        total = sum(sc.values()) or 1
        pct = sc[best] / total * 100
        print(f"  {delivery:<22} {best:<22} {pct:>5.1f}%")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None)
    p.add_argument(
        "--agent", type=str, default="qlearning", choices=["qlearning", "dqn"]
    )
    p.add_argument("--format", type=str, default="T20", choices=["T20", "ODI", "Test"])
    p.add_argument("--episodes", type=int, default=1_000)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = CHECKPOINT_DIR / f"{args.agent}_best.pkl"
        if not model_path.exists():
            model_path = CHECKPOINT_DIR / f"{args.agent}_final.pkl"

    if not model_path.exists():
        print(f"[Visualizer] No model at {model_path}. Train first.")
        sys.exit(1)

    AgentCls = QLearningAgent if args.agent == "qlearning" else DQNAgent
    agent = AgentCls.load(str(model_path))
    agent.epsilon = 0.0

    print(f"[Visualizer] Collecting data from {args.episodes} greedy episodes ...")
    data = collect_shot_data(agent, args.format, args.episodes)

    print_strategy_table(data)
    plot_shot_delivery_heatmap(data, PLOT_DIR / "shot_delivery_heatmap.png")
    plot_preferred_shots_bar(data, PLOT_DIR / "preferred_shots_bar.png")
    plot_pressure_heatmap(agent, args.format, PLOT_DIR / "pressure_heatmap.png")

    print(f"\n[Visualizer] All plots saved to {PLOT_DIR}/")
