"""
training/train_agent.py
------------------------
Training loop supporting both Q-learning and DQN agents.

Usage
-----
    python training/train_agent.py                      # Q-learning, 10 000 eps
    python training/train_agent.py --agent dqn          # DQN
    python training/train_agent.py --episodes 50000     # custom count
    python training/train_agent.py --format ODI         # ODI format
    python training/train_agent.py --skill 0.8          # high-skill batter
    python training/train_agent.py --resume             # continue from checkpoint
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agents.bowler_agent import BowlerAgent
from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from environment.cricket_env import CricketEnv
from environment.multi_agent_env import MultiAgentCricketEnv
from utils import logger as match_logger

TOTAL_BALLS_MAP = {"T20": 120, "ODI": 300, "Test": 450}
SAVE_EVERY = 2_000
LOG_EVERY = 500
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def train(
    episodes: int = 10_000,
    match_format: str = "T20",
    agent_type: str = "qlearning",
    batter_skill: float = 0.5,
    resume: bool = False,
    verbose: bool = False,
) -> object:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    total_balls = TOTAL_BALLS_MAP.get(match_format.upper(), 120)
    is_multi = agent_type.lower() == "multi"
    env = (
        MultiAgentCricketEnv(total_balls=total_balls, match_format=match_format)
        if is_multi
        else CricketEnv(
            total_balls=total_balls,
            match_format=match_format,
            batter_skill=batter_skill,
            verbose=verbose,
        )
    )
    n_actions = 14 if not is_multi else None  # for single agent

    agent_label = agent_type.lower()
    if is_multi:
        batter_final_path = CHECKPOINT_DIR / "multi_batter_final.pkl"
        bowler_final_path = CHECKPOINT_DIR / "multi_bowler_final.pkl"
        batter_best_path = CHECKPOINT_DIR / "multi_batter_best.pkl"
        bowler_best_path = CHECKPOINT_DIR / "multi_bowler_best.pkl"
    else:
        final_path = CHECKPOINT_DIR / f"{agent_label}_final.pkl"
        best_path = CHECKPOINT_DIR / f"{agent_label}_best.pkl"

    if is_multi:
        batter_agent = QLearningAgent(n_actions=14)
        bowler_agent = BowlerAgent(n_actions=11)
        agent = (batter_agent, bowler_agent)  # tuple
        print(f"[Train] Starting fresh multi-agent training.")
    else:
        if resume and final_path.exists():
            agent = (
                QLearningAgent.load if agent_label == "qlearning" else DQNAgent.load
            )(str(final_path))
            print(f"[Train] Resuming {agent_label} from checkpoint.")
        else:
            agent = (
                QLearningAgent(n_actions=n_actions)
                if agent_label == "qlearning"
                else DQNAgent(n_actions=n_actions)
            )
            print(f"[Train] Starting fresh {agent_label} training.")

    print(
        f"[Train] Format={match_format}  Episodes={episodes:,}  "
        f"Skill={batter_skill}  Agent={agent_label}"
    )
    print("-" * 65)

    reward_window = deque(maxlen=500)
    score_window = deque(maxlen=500)
    wickets_window = deque(maxlen=500)
    best_avg = -float("inf")
    start = time.time()

    for ep in range(1, episodes + 1):
        if is_multi:
            batter_state, bowler_state = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                batter_avail = env.batter_actions()
                bowler_avail = env.bowler_actions()
                ba = batter_agent.choose_action(batter_state, batter_avail)
                bwa = bowler_agent.choose_action(bowler_state, bowler_avail)
                (b_state, bw_state), (b_reward, bw_reward), done, info = env.step(
                    ba, bwa
                )
                next_batter_avail = env.batter_actions() if not done else []
                next_bowler_avail = env.bowler_actions() if not done else []
                batter_agent.update(
                    batter_state, ba, b_reward, b_state, done, next_batter_avail
                )
                bowler_agent.update(
                    bowler_state, bwa, bw_reward, bw_state, done, next_bowler_avail
                )
                batter_state = b_state
                bowler_state = bw_state
                ep_reward += b_reward
                if verbose:
                    env.render()
        else:
            state = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                avail = env.get_available_actions()
                action = agent.choose_action(state, avail)
                next_state, reward, done, info = env.step(action)
                next_avail = env.get_available_actions() if not done else []
                agent.update(state, action, reward, next_state, done, next_avail)
                state = next_state
                ep_reward += reward
                if verbose:
                    env.render()

        reward_window.append(ep_reward)
        score_window.append(env.score)
        wickets_window.append(env.wickets)

        # Log every completed innings to match history
        if ep % 100 == 0:
            match_logger.log_innings(
                match_format=match_format,
                innings=1,
                score=env.score,
                wickets=env.wickets,
                balls_played=env.balls_bowled,
                agent_type=agent_label,
                extra={"episode": ep, "reward": round(ep_reward, 2)},
            )

        if ep % LOG_EVERY == 0:
            avg_r = np.mean(reward_window)
            avg_s = np.mean(score_window)
            avg_w = np.mean(wickets_window)
            eps_ps = ep / (time.time() - start)
            eps_str = (
                f"{getattr(batter_agent if is_multi else agent, 'epsilon', 0):.4f}"
            )
            print(
                f"Ep {ep:>7,}/{episodes:,}  "
                f"AvgReward: {avg_r:>8.2f}  "
                f"AvgScore: {avg_s:>6.1f}  "
                f"AvgWkts: {avg_w:>4.1f}  "
                f"ε: {eps_str}  "
                f"{eps_ps:.0f} ep/s"
            )

        if ep % SAVE_EVERY == 0:
            if is_multi:
                batter_ckpt = CHECKPOINT_DIR / f"multi_batter_ep{ep}.pkl"
                bowler_ckpt = CHECKPOINT_DIR / f"multi_bowler_ep{ep}.pkl"
                batter_agent.save(str(batter_ckpt))
                bowler_agent.save(str(bowler_ckpt))
            else:
                ckpt = CHECKPOINT_DIR / f"{agent_label}_ep{ep}.pkl"
                agent.save(str(ckpt))
            avg_r = np.mean(reward_window)
            if avg_r > best_avg:
                best_avg = avg_r
                if is_multi:
                    batter_agent.save(str(batter_best_path))
                    bowler_agent.save(str(bowler_best_path))
                else:
                    agent.save(str(best_path))

    if is_multi:
        batter_agent.save(str(batter_final_path))
        bowler_agent.save(str(bowler_final_path))
    else:
        agent.save(str(final_path))
    elapsed = time.time() - start
    print("-" * 65)
    print(f"[Train] Done in {elapsed:.1f}s  Best avg reward: {best_avg:.2f}")
    print(f"[Train] Summary: {match_logger.summarise_logs()}")
    return agent


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10_000)
    p.add_argument("--format", type=str, default="T20", choices=["T20", "ODI", "Test"])
    p.add_argument(
        "--agent", type=str, default="qlearning", choices=["qlearning", "dqn", "multi"]
    )
    p.add_argument("--skill", type=float, default=0.5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    train(a.episodes, a.format, a.agent, a.skill, a.resume, a.verbose)
