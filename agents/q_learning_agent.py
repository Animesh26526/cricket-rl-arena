"""
agents/q_learning_agent.py
---------------------------
Tabular Q-learning agent for cricket shot selection (v2).

State discretisation updated to match the new context-aware state vector:
  [delivery_idx, stumps_idx, wickets_remaining, balls_remaining,
   runs_required, current_run_rate, required_run_rate]

Q-table shape:
  (11 deliveries × 2 stumps × 11 wickets × 6 ball_bins ×
   5 rrr_bins × 4 crr_bins × 5 pressure_bins)  × n_actions
"""

import random
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np


DEFAULTS = dict(
    alpha         = 0.1,
    gamma         = 0.95,
    epsilon       = 1.0,
    epsilon_min   = 0.05,
    epsilon_decay = 0.9995,
    n_deliveries  = 11,
    n_stumps      = 2,
    n_wickets     = 11,
    n_balls_bins  = 6,
    n_rrr_bins    = 5,    # required run rate buckets
    n_crr_bins    = 4,    # current run rate buckets
    n_pressure    = 5,    # runs_required buckets
    n_actions     = 14,
)


class QLearningAgent:
    """
    ε-greedy tabular Q-learning agent.

    Compatible with both the original CricketEnv and the new context-aware
    version — the state vector is always length-7.
    """

    def __init__(self, n_actions: int = DEFAULTS["n_actions"], **kwargs):
        cfg = {**DEFAULTS, **kwargs, "n_actions": n_actions}

        self.alpha         = cfg["alpha"]
        self.gamma         = cfg["gamma"]
        self.epsilon       = cfg["epsilon"]
        self.epsilon_min   = cfg["epsilon_min"]
        self.epsilon_decay = cfg["epsilon_decay"]
        self.n_actions     = n_actions

        self._n_del  = cfg["n_deliveries"]
        self._n_st   = cfg["n_stumps"]
        self._n_wk   = cfg["n_wickets"]
        self._n_bl   = cfg["n_balls_bins"]
        self._n_rrr  = cfg["n_rrr_bins"]
        self._n_crr  = cfg["n_crr_bins"]
        self._n_pr   = cfg["n_pressure"]

        shape = (
            self._n_del, self._n_st, self._n_wk,
            self._n_bl, self._n_rrr, self._n_crr,
            self._n_pr, self.n_actions,
        )
        self.q_table: np.ndarray = np.zeros(shape, dtype=np.float32)

        self.total_steps  = 0
        self.total_reward = 0.0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state: List[float], available_actions: List[int]) -> int:
        if not available_actions:
            return 0
        # ε-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        s     = self._discretise(state)
        q_row = self.q_table[s]
        # pick uniformly among the best actions so behaviour is less deterministic
        best_val = max(q_row[a] for a in available_actions if a < len(q_row))
        best_actions = [a for a in available_actions if a < len(q_row) and q_row[a] == best_val]
        return random.choice(best_actions)

    def greedy_action(self, state: List[float], available_actions: List[int]) -> int:
        """Pure greedy — used during evaluation but still randomize ties."""
        if not available_actions:
            return 0
        s     = self._discretise(state)
        q_row = self.q_table[s]
        best_val = max(q_row[a] for a in available_actions if a < len(q_row))
        best_actions = [a for a in available_actions if a < len(q_row) and q_row[a] == best_val]
        return random.choice(best_actions)

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def update(
        self,
        state:          List[float],
        action:         int,
        reward:         float,
        next_state:     List[float],
        done:           bool,
        available_next: List[int],
    ) -> None:
        if action >= self.n_actions:
            return
        s  = self._discretise(state)
        s2 = self._discretise(next_state)

        current_q = self.q_table[s][action]
        if done or not available_next:
            target_q = reward
        else:
            best_next = max(
                available_next,
                key=lambda a: self.q_table[s2][a] if a < self.n_actions else -999,
            )
            target_q = reward + self.gamma * self.q_table[s2][best_next]

        self.q_table[s][action] += self.alpha * (target_q - current_q)
        self.total_steps  += 1
        self.total_reward += reward
        self._decay_epsilon()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"[QLearningAgent] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(n_actions=data.get("n_actions", DEFAULTS["n_actions"]))
        agent.__dict__.update(data)
        print(f"[QLearningAgent] Loaded ← {path}  (ε={agent.epsilon:.4f})")
        return agent

    # ------------------------------------------------------------------
    # Private: state discretisation (updated for v2 state)
    # ------------------------------------------------------------------

    def _discretise(self, state: List[float]) -> Tuple:
        """
        Map 7-feature raw state → discrete Q-table index.

        state = [delivery_idx, stumps_idx, wickets_remaining,
                 balls_remaining, runs_required, current_rr, required_rr]
        """
        d_idx, st_idx, wk, balls_rem, runs_req, crr, rrr = state

        # delivery (0-10)
        d  = int(min(d_idx,  self._n_del - 1))
        # stumps (0-1)
        st = int(min(st_idx, self._n_st  - 1))
        # wickets remaining (0-10)
        wk_b = int(min(wk, self._n_wk - 1))
        # balls remaining → 6 buckets  (0-20, 20-40, …, 100+)
        bl_b = min(int(balls_rem // 20), self._n_bl - 1)
        # required run rate → 5 buckets (0-6, 6-8, 8-10, 10-14, 14+)
        rrr_thresholds = [6, 8, 10, 14]
        rrr_b = sum(1 for t in rrr_thresholds if rrr > t)
        rrr_b = min(rrr_b, self._n_rrr - 1)
        # current run rate → 4 buckets (0-5, 5-8, 8-11, 11+)
        crr_thresholds = [5, 8, 11]
        crr_b = sum(1 for t in crr_thresholds if crr > t)
        crr_b = min(crr_b, self._n_crr - 1)
        # runs required → 5 buckets (0, 1-30, 31-60, 61-100, 100+)
        pr_thresholds = [1, 31, 61, 101]
        pr_b = sum(1 for t in pr_thresholds if runs_req >= t)
        pr_b = min(pr_b, self._n_pr - 1)

        return (d, st, wk_b, bl_b, rrr_b, crr_b, pr_b)

    def _decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
