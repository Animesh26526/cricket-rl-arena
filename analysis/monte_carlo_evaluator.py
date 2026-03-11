"""
analysis/monte_carlo_evaluator.py
-----------------------------------
Monte Carlo match simulator and policy evaluator.

Runs thousands of simulated innings using a trained agent, collects
statistics, and generates publication-quality visualisations.

Outputs (saved to analysis/plots/)
-------
  mc_score_histogram.png       — distribution of total scores
  mc_shot_heatmap.png          — shot selection by delivery (frequency)
  mc_win_probability.png       — P(win) vs runs required to chase
  mc_wicket_distribution.png   — distribution of wickets lost per innings

Usage
-----
    python analysis/monte_carlo_evaluator.py                    # 10 000 sims
    python analysis/monte_carlo_evaluator.py --sims 50000 --format ODI
    python analysis/monte_carlo_evaluator.py --agent dqn
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from environment.cricket_env import CricketEnv
from environment.probability_engine import ALL_DELIVERIES

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "training" / "checkpoints"
PLOT_DIR = Path(__file__).resolve().parent / "plots"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOTAL_BALLS_MAP = {"T20": 120, "ODI": 300, "Test": 450}


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def run_simulations(
    agent,
    match_format: str,
    n_sims: int,
    target: Optional[int] = None,
    batter_skill: float = 0.5,
    verbose_every: int = 1000,
) -> Dict:
    """
    Run `n_sims` greedy innings and collect comprehensive statistics.

    Returns
    -------
    dict with keys: scores, wickets, balls_played, shot_counts,
                    win_results, delivery_counts
    """
    total_balls = TOTAL_BALLS_MAP.get(match_format.upper(), 120)
    env = CricketEnv(
        total_balls=total_balls,
        match_format=match_format,
        target=target,
        batter_skill=batter_skill,
    )

    scores: List[int] = []
    wickets_list: List[int] = []
    balls_list: List[int] = []
    wins: List[bool] = []

    shot_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    delivery_counts: Dict[str, int] = defaultdict(int)

    t0 = time.time()
    for i in range(1, n_sims + 1):
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
            state, _, done, _ = env.step(action)
            shot_counts[delivery][shot] += 1
            delivery_counts[delivery] += 1

        scores.append(env.score)
        wickets_list.append(env.wickets)
        balls_list.append(env.balls_bowled)
        if target is not None:
            wins.append(env.score >= target)

        if verbose_every and i % verbose_every == 0:
            elapsed = time.time() - t0
            avg_score = np.mean(scores[-verbose_every:])
            print(
                f"  [{i:>6,}/{n_sims:,}]  Avg score (last {verbose_every}): {avg_score:.1f}  "
                f"{i/elapsed:.0f} sims/s"
            )

    return {
        "scores": scores,
        "wickets": wickets_list,
        "balls_played": balls_list,
        "shot_counts": {k: dict(v) for k, v in shot_counts.items()},
        "delivery_counts": dict(delivery_counts),
        "wins": wins,
        "target": target,
        "match_format": match_format,
        "n_sims": n_sims,
        "batter_skill": batter_skill,
    }


# ---------------------------------------------------------------------------
# Statistics summary
# ---------------------------------------------------------------------------


def compute_summary(results: Dict) -> Dict:
    scores = np.array(results["scores"])
    wickets = np.array(results["wickets"])
    balls = np.array(results["balls_played"])
    wins = results["wins"]
    overs = balls / 6.0

    summary = {
        "n_sims": results["n_sims"],
        "format": results["match_format"],
        "skill": results["batter_skill"],
        "avg_score": round(float(np.mean(scores)), 2),
        "std_score": round(float(np.std(scores)), 2),
        "median_score": int(np.median(scores)),
        "p10_score": int(np.percentile(scores, 10)),
        "p90_score": int(np.percentile(scores, 90)),
        "avg_wickets": round(float(np.mean(wickets)), 2),
        "avg_balls": round(float(np.mean(balls)), 2),
        "avg_run_rate": round(float(np.mean(scores / np.maximum(overs, 0.01))), 2),
        "win_probability": round(sum(wins) / len(wins), 4) if wins else None,
        "top_delivery": (
            max(results["delivery_counts"], key=results["delivery_counts"].get)
            if results["delivery_counts"]
            else "N/A"
        ),
    }

    # Top shot overall
    all_shots: Dict[str, int] = defaultdict(int)
    for delivery_shots in results["shot_counts"].values():
        for shot, cnt in delivery_shots.items():
            all_shots[shot] += cnt
    summary["top_shot"] = max(all_shots, key=all_shots.get) if all_shots else "N/A"

    return summary


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def plot_score_histogram(results: Dict, save_path: Path) -> None:
    scores = results["scores"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores, bins=40, color="#2563eb", edgecolor="white", alpha=0.85)
    ax.axvline(
        np.mean(scores),
        color="#ef4444",
        lw=2,
        linestyle="--",
        label=f"Mean: {np.mean(scores):.1f}",
    )
    ax.axvline(
        np.median(scores),
        color="#f59e0b",
        lw=2,
        linestyle=":",
        label=f"Median: {int(np.median(scores))}",
    )
    ax.set_xlabel("Total Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        f"Score Distribution — {results['match_format']} ({results['n_sims']:,} simulations)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[MC] Saved: {save_path}")


def plot_shot_heatmap(results: Dict, save_path: Path) -> None:
    shot_counts = results["shot_counts"]
    all_shots = sorted({s for d in shot_counts.values() for s in d})
    matrix = np.zeros((len(ALL_DELIVERIES), len(all_shots)))
    for di, delivery in enumerate(ALL_DELIVERIES):
        sc = shot_counts.get(delivery, {})
        total = sum(sc.values()) or 1
        for si, shot in enumerate(all_shots):
            matrix[di, si] = sc.get(shot, 0) / total * 100

    fig, ax = plt.subplots(figsize=(max(10, len(all_shots)), 7))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=all_shots,
        yticklabels=ALL_DELIVERIES,
        annot=True,
        fmt=".0f",
        cmap="rocket_r",
        linewidths=0.3,
        linecolor="#e5e7eb",
        cbar_kws={"label": "Frequency (%)"},
    )
    ax.set_title(
        f"Shot Selection Heatmap — {results['match_format']} ({results['n_sims']:,} sims)",
        fontsize=12,
    )
    ax.set_xlabel("Shot")
    ax.set_ylabel("Delivery")
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[MC] Saved: {save_path}")


def plot_win_probability(
    match_format: str, agent, save_path: Path, skill: float = 0.5
) -> None:
    """Sweep different targets and plot P(win) vs runs required."""
    total_balls = TOTAL_BALLS_MAP.get(match_format, 120)
    targets = list(range(50, 230, 15))
    win_probs = []
    n_per = 500

    print("[MC] Computing win probabilities ...")
    for target in targets:
        env = CricketEnv(
            total_balls=total_balls,
            match_format=match_format,
            target=target,
            batter_skill=skill,
        )
        wins = 0
        for _ in range(n_per):
            state = env.reset()
            done = False
            while not done:
                avail = env.get_available_actions()
                action = agent.greedy_action(state, avail)
                state, _, done, _ = env.step(action)
            if env.score >= target:
                wins += 1
        win_probs.append(wins / n_per)
        print(f"  Target {target}: P(win)={wins/n_per:.3f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(targets, [p * 100 for p in win_probs], "o-", color="#2563eb", lw=2.5, ms=6)
    ax.fill_between(targets, [p * 100 for p in win_probs], alpha=0.15, color="#2563eb")
    ax.axhline(50, color="#9ca3af", lw=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Target (runs to chase)", fontsize=12)
    ax.set_ylabel("Win Probability (%)", fontsize=12)
    ax.set_title(f"Win Probability vs Target — {match_format}", fontsize=13)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[MC] Saved: {save_path}")


def plot_wicket_distribution(results: Dict, save_path: Path) -> None:
    wickets = results["wickets"]
    unique, counts = np.unique(wickets, return_counts=True)
    pct = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(unique, pct, color="#10b981", edgecolor="white", width=0.65)
    for bar, p in zip(bars, pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{p:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xlabel("Wickets Lost", fontsize=12)
    ax.set_ylabel("Frequency (%)", fontsize=12)
    ax.set_title(
        f"Wickets per Innings — {results['match_format']} ({results['n_sims']:,} sims)",
        fontsize=13,
    )
    ax.set_xticks(range(0, 11))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[MC] Saved: {save_path}")


# ---------------------------------------------------------------------------
# Save results to JSON
# ---------------------------------------------------------------------------


def save_results(results: Dict, summary: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "score_sample": results["scores"][:200],  # first 200 for brevity
        "shot_counts": results["shot_counts"],
        "delivery_counts": results["delivery_counts"],
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[MC] Results saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--sims", type=int, default=10_000)
    p.add_argument("--format", type=str, default="T20", choices=["T20", "ODI", "Test"])
    p.add_argument(
        "--agent", type=str, default="qlearning", choices=["qlearning", "dqn"]
    )
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--target", type=int, default=None)
    p.add_argument("--skill", type=float, default=0.5)
    p.add_argument(
        "--no-win-curve",
        action="store_true",
        help="Skip the win-probability sweep (saves time)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = CHECKPOINT_DIR / f"{args.agent}_best.pkl"
        if not model_path.exists():
            model_path = CHECKPOINT_DIR / f"{args.agent}_final.pkl"

    if not model_path.exists():
        print(f"[MC] No model at {model_path}. Train first.")
        sys.exit(1)

    AgentCls = QLearningAgent if args.agent == "qlearning" else DQNAgent
    agent = AgentCls.load(str(model_path))
    agent.epsilon = 0.0

    print(
        f"\n[MC] Running {args.sims:,} simulations  format={args.format}  skill={args.skill}"
    )
    print("-" * 60)
    results = run_simulations(
        agent,
        args.format,
        args.sims,
        target=args.target,
        batter_skill=args.skill,
    )
    summary = compute_summary(results)

    # --- Print summary ---
    print("\n" + "=" * 55)
    print(f"  MONTE CARLO SUMMARY  |  {args.format}  |  {args.sims:,} sims")
    print("=" * 55)
    for k, v in summary.items():
        print(f"  {k:<25}: {v}")

    # --- Save plots ---
    plot_score_histogram(results, PLOT_DIR / "mc_score_histogram.png")
    plot_shot_heatmap(results, PLOT_DIR / "mc_shot_heatmap.png")
    plot_wicket_distribution(results, PLOT_DIR / "mc_wicket_distribution.png")

    if not args.no_win_curve:
        plot_win_probability(
            args.format, agent, PLOT_DIR / "mc_win_probability.png", skill=args.skill
        )

    save_results(
        results, summary, RESULTS_DIR / f"mc_results_{args.format.lower()}.json"
    )
    print(f"\n[MC] All outputs saved to {PLOT_DIR}/ and {RESULTS_DIR}/")
