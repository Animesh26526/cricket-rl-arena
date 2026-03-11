"""
agents/bowler_agent.py
-----------------------
Q-learning agent that controls delivery selection (bowling side).

Bowler state (6 features from MultiAgentCricketEnv.get_bowler_state()):
  [wickets_taken, balls_bowled_bucket, runs_conceded_bucket,
   current_economy, innings_phase, pressure_flag]

Actions: index into ALL_DELIVERIES  (0–10, 11 deliveries total)

Reward: negative of batter reward (zero-sum) — the bowler earns:
  +15 per wicket, +0.5 per dot ball, penalty per run conceded above threshold.
"""

import random
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

DEFAULTS = dict(
    alpha=0.1,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.9995,
    n_deliveries=11,  # action space = ALL_DELIVERIES
    # state discretisation bins
    n_wickets=11,
    n_balls_bins=6,
    n_score_bins=6,
    n_economy=4,
    n_phase=4,
    n_pressure=2,
)


class BowlerAgent:
    """
    Tabular ε-greedy Q-learning agent for bowling delivery selection.

    Learns which deliveries are most effective in each match situation.
    """

    def __init__(self, n_actions: int = 11, **kwargs):
        cfg = {**DEFAULTS, **kwargs, "n_deliveries": n_actions}

        self.alpha = cfg["alpha"]
        self.gamma = cfg["gamma"]
        self.epsilon = cfg["epsilon"]
        self.epsilon_min = cfg["epsilon_min"]
        self.epsilon_decay = cfg["epsilon_decay"]
        self.n_actions = n_actions

        self._n_wk = cfg["n_wickets"]
        self._n_bl = cfg["n_balls_bins"]
        self._n_sc = cfg["n_score_bins"]
        self._n_ec = cfg["n_economy"]
        self._n_ph = cfg["n_phase"]
        self._n_pr = cfg["n_pressure"]

        shape = (
            self._n_wk,
            self._n_bl,
            self._n_sc,
            self._n_ec,
            self._n_ph,
            self._n_pr,
            self.n_actions,
        )
        self.q_table: np.ndarray = np.zeros(shape, dtype=np.float32)
        self.total_steps = 0
        self.total_reward = 0.0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state: List[float], available_actions: List[int]) -> int:
        if not available_actions:
            return 0
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        s = self._discretise(state)
        q = self.q_table[s]
        return max(available_actions, key=lambda a: q[a] if a < len(q) else -999)

    def greedy_action(self, state: List[float], available_actions: List[int]) -> int:
        if not available_actions:
            return 0
        s = self._discretise(state)
        q = self.q_table[s]
        return max(available_actions, key=lambda a: q[a] if a < len(q) else -999)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        state: List[float],
        action: int,
        reward: float,
        next_state: List[float],
        done: bool,
        available_next: List[int],
    ) -> None:
        if action >= self.n_actions:
            return
        s = self._discretise(state)
        s2 = self._discretise(next_state)

        current_q = self.q_table[s][action]
        if done or not available_next:
            target_q = reward
        else:
            best = max(
                available_next,
                key=lambda a: self.q_table[s2][a] if a < self.n_actions else -999,
            )
            target_q = reward + self.gamma * self.q_table[s2][best]

        self.q_table[s][action] += self.alpha * (target_q - current_q)
        self.total_steps += 1
        self.total_reward += reward
        self._decay_epsilon()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"[BowlerAgent] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "BowlerAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(n_actions=data.get("n_actions", 11))
        agent.__dict__.update(data)
        print(f"[BowlerAgent] Loaded ← {path}  (ε={agent.epsilon:.4f})")
        return agent

    # ------------------------------------------------------------------
    # Private: discretisation
    # ------------------------------------------------------------------

    def _discretise(self, state: List[float]) -> Tuple:
        """
        state = [wickets_taken, balls_bucket, score_bucket,
                 economy, phase, pressure_flag]
        """
        wickets, bl, sc, eco, phase, pressure = state

        wk = int(min(wickets, self._n_wk - 1))
        bl_ = int(min(bl, self._n_bl - 1))
        sc_ = int(min(sc, self._n_sc - 1))
        # economy → 4 buckets: 0-6, 6-9, 9-12, 12+
        ec_ = min(int(eco / 3), self._n_ec - 1)
        ph_ = int(min(phase, self._n_ph - 1))
        pr_ = int(min(pressure, self._n_pr - 1))
        return (wk, bl_, sc_, ec_, ph_, pr_)

    def _decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
