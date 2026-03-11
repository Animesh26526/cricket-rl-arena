"""
training/evaluate_agent.py
---------------------------
Evaluate a trained Q-learning agent and display its learned shot preferences.

Usage
-----
    python training/evaluate_agent.py                     # evaluate best checkpoint
    python training/evaluate_agent.py --model path/to/model.pkl
    python training/evaluate_agent.py --episodes 200 --format ODI

Output
------
- Average score / wickets / reward over N evaluation episodes.
- A per-delivery shot preference table showing which shot the agent
  most prefers against each delivery type (stumps Touching and Not Touching).
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agents.q_learning_agent import QLearningAgent
from environment.cricket_env import CricketEnv
from environment.probability_engine import ALL_DELIVERIES

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_MODEL = CHECKPOINT_DIR / "q_agent_best.pkl"

TOTAL_BALLS_MAP = {"T20": 120, "ODI": 300, "Test": 450}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def run_evaluation(
    agent: QLearningAgent,
    episodes: int,
    match_format: str,
) -> Dict:
    """Run `episodes` greedy innings and return aggregate statistics."""
    total_balls = TOTAL_BALLS_MAP.get(match_format.upper(), 120)
    env = CricketEnv(total_balls=total_balls, match_format=match_format)

    scores, wickets_list, rewards = [], [], []
    shot_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            available = env.get_available_actions()
            action = agent.greedy_action(state, available)
            shot = (
                env.available_shots[action]
                if action < len(env.available_shots)
                else "Unknown"
            )
            delivery = env.delivery

            state, reward, done, info = env.step(action)
            ep_reward += reward
            shot_counts[delivery][shot] += 1

        scores.append(env.score)
        wickets_list.append(env.wickets)
        rewards.append(ep_reward)

    return {
        "episodes": episodes,
        "avg_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "avg_wickets": float(np.mean(wickets_list)),
        "avg_reward": float(np.mean(rewards)),
        "shot_counts": dict(shot_counts),
        "scores": scores,
    }


def print_summary(stats: Dict, match_format: str) -> None:
    """Print a nicely formatted evaluation summary."""
    width = 62
    print("\n" + "=" * width)
    print(
        f"  AGENT EVALUATION  |  Format: {match_format}  |  Episodes: {stats['episodes']}"
    )
    print("=" * width)
    print(f"  Avg Score   : {stats['avg_score']:.1f}  (σ = {stats['std_score']:.1f})")
    print(f"  Avg Wickets : {stats['avg_wickets']:.2f}")
    print(f"  Avg Reward  : {stats['avg_reward']:.2f}")
    print("=" * width)


def print_shot_preferences(shot_counts: Dict, top_n: int = 3) -> None:
    """Print which shots the agent prefers against each delivery."""
    print("\n  LEARNED SHOT PREFERENCES (top 3 per delivery)")
    print("-" * 62)
    header = f"  {'Delivery':<22} {'Shot':<22} {'Frequency':>10}"
    print(header)
    print("-" * 62)

    for delivery in ALL_DELIVERIES:
        counts = shot_counts.get(delivery, {})
        if not counts:
            continue
        total = sum(counts.values())
        sorted_shots = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        for i, (shot, count) in enumerate(sorted_shots):
            pct = count / total * 100
            label = delivery if i == 0 else ""
            print(f"  {label:<22} {shot:<22} {pct:>9.1f}%")
        print()


def score_histogram(scores, bins=10) -> None:
    """Print a simple ASCII histogram of scores."""
    if not scores:
        return
    min_s, max_s = min(scores), max(scores)
    bin_width = max(1, (max_s - min_s) // bins)
    print("  SCORE DISTRIBUTION")
    print("-" * 42)
    bucket_counts: Dict[int, int] = defaultdict(int)
    for s in scores:
        bucket = (s - min_s) // bin_width
        bucket_counts[bucket] += 1
    max_count = max(bucket_counts.values()) if bucket_counts else 1
    bar_max = 30

    for i in range(bins):
        lo = min_s + i * bin_width
        hi = lo + bin_width
        count = bucket_counts.get(i, 0)
        bar = "█" * int(count / max_count * bar_max)
        print(f"  {lo:>4}-{hi:<4} | {bar:<{bar_max}} {count}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained cricket agent.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to saved agent pickle",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of evaluation episodes (default: 500)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="T20",
        choices=["T20", "ODI", "Test"],
        help="Match format (default: T20)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Top N shots to show per delivery (default: 3)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"[Evaluate] No model found at {model_path}.")
        print("[Evaluate] Train first with:  python training/train_agent.py")
        sys.exit(1)

    print(f"[Evaluate] Loading model from {model_path} ...")
    agent = QLearningAgent.load(str(model_path))
    # Switch off exploration for evaluation
    agent.epsilon = 0.0

    print(f"[Evaluate] Running {args.episodes} greedy episodes ({args.format}) ...")
    stats = run_evaluation(agent, args.episodes, args.format)

    print_summary(stats, args.format)
    print_shot_preferences(stats["shot_counts"], top_n=args.top)
    score_histogram(stats["scores"])
