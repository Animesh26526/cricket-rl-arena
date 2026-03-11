"""
agents/dqn_agent.py
--------------------
Deep Q-Network (DQN) agent implemented with pure NumPy.

Architecture
------------
  Input  : 7-feature normalised state vector
  Hidden : Linear(7→128) → ReLU → Linear(128→64) → ReLU
  Output : Linear(64→n_actions)  [one Q-value per action]

Features
--------
  • Experience replay buffer (random mini-batch sampling)
  • Separate target network, hard-updated every C steps
  • ε-greedy exploration with exponential decay
  • Save / load via pickle
  • Compatible with any environment that exposes the same
    reset() / step() / get_available_actions() interface as CricketEnv.

Usage
-----
    from agents.dqn_agent import DQNAgent
    agent = DQNAgent(state_size=7, n_actions=14)
    # training loop identical to QLearningAgent except .update() signature
"""

import random
import pickle
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# State normalisation constants (per-feature min/max for [0,1] scaling)
# ---------------------------------------------------------------------------
STATE_MIN = np.array([0., 0., 0.,   0.,  0.,  0.,  0.], dtype=np.float32)
STATE_MAX = np.array([10., 1., 10., 300., 500., 36., 36.], dtype=np.float32)


def normalise_state(state: List[float]) -> np.ndarray:
    s = np.array(state, dtype=np.float32)
    rng = np.where(STATE_MAX > STATE_MIN, STATE_MAX - STATE_MIN, 1.0)
    return np.clip((s - STATE_MIN) / rng, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Tiny neural-net layer helpers (pure NumPy)
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Experience Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Circular buffer that stores (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int = 50_000):
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self._buf.append((state, action, float(reward), next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.vstack(states).astype(np.float32),
            np.array(actions,  dtype=np.int32),
            np.array(rewards,  dtype=np.float32),
            np.vstack(next_states).astype(np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# QNetwork (two hidden layers, pure NumPy)
# ---------------------------------------------------------------------------

class QNetwork:
    """
    Three-layer fully connected network.

      7 → 128 → 64 → n_actions

    Parameters stored as weight matrices and bias vectors.
    Trained via SGD with Huber-like gradient clipping.
    """

    def __init__(self, state_size: int, n_actions: int, lr: float = 1e-3):
        self.lr = lr
        # Xavier initialisation
        def xavier(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

        self.W1 = xavier(state_size, 128)
        self.b1 = np.zeros((1, 128), dtype=np.float32)
        self.W2 = xavier(128, 64)
        self.b2 = np.zeros((1, 64),  dtype=np.float32)
        self.W3 = xavier(64, n_actions)
        self.b3 = np.zeros((1, n_actions), dtype=np.float32)

        # Adam optimiser state
        self._t = 0
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Forward pass; returns Q-values and cache for backward."""
        z1 = X  @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        q  = a2 @ self.W3 + self.b3
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "q": q}
        return q, cache

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict Q-values for a single state (no gradient needed)."""
        q, _ = self.forward(state.reshape(1, -1))
        return q[0]

    def train_step(
        self,
        states:      np.ndarray,
        actions:     np.ndarray,
        targets:     np.ndarray,
    ) -> float:
        """
        One mini-batch SGD step.

        Only the Q-value at the taken action index is trained;
        other actions have zero gradient (standard DQN).
        """
        q_pred, cache = self.forward(states)

        # Loss: MSE on selected actions only
        errors = np.zeros_like(q_pred)
        batch_idx = np.arange(len(actions))
        errors[batch_idx, actions] = q_pred[batch_idx, actions] - targets
        # Huber-style clipping
        errors = np.clip(errors, -1.0, 1.0)

        loss = float(np.mean(errors[batch_idx, actions] ** 2))

        # Backprop
        dq  = errors / len(actions)
        dW3 = cache["a2"].T @ dq
        db3 = dq.sum(axis=0, keepdims=True)

        da2 = dq @ self.W3.T
        dz2 = da2 * relu_grad(cache["z2"])
        dW2 = cache["a1"].T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(cache["z1"])
        dW1 = cache["X"].T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        grads = [dW1, db1, dW2, db2, dW3, db3]
        self._adam_update(grads)
        return loss

    # Adam optimiser
    def _adam_update(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        params = self._params()
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * g ** 2
            m_hat = self._m[i] / (1 - beta1 ** self._t)
            v_hat = self._v[i] / (1 - beta2 ** self._t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def copy_weights_from(self, other: "QNetwork") -> None:
        """Hard-copy weights from another QNetwork (target-network update)."""
        self.W1[:] = other.W1
        self.b1[:] = other.b1
        self.W2[:] = other.W2
        self.b2[:] = other.b2
        self.W3[:] = other.W3
        self.b3[:] = other.b3


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.

    Parameters
    ----------
    state_size      : int    dimension of state vector (7)
    n_actions       : int    maximum number of shots (14)
    lr              : float  learning rate for Adam  (1e-3)
    gamma           : float  discount factor          (0.95)
    epsilon         : float  initial exploration rate (1.0)
    epsilon_min     : float  minimum epsilon          (0.05)
    epsilon_decay   : float  multiplicative decay     (0.9995)
    buffer_capacity : int    replay buffer size       (50 000)
    batch_size      : int    mini-batch size          (64)
    target_update   : int    hard update every N steps (500)
    """

    def __init__(
        self,
        state_size:      int   = 7,
        n_actions:       int   = 14,
        lr:              float = 1e-3,
        gamma:           float = 0.95,
        epsilon:         float = 1.0,
        epsilon_min:     float = 0.05,
        epsilon_decay:   float = 0.9995,
        buffer_capacity: int   = 50_000,
        batch_size:      int   = 64,
        target_update:   int   = 500,
    ):
        self.state_size    = state_size
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update

        self.online_net  = QNetwork(state_size, n_actions, lr)
        self.target_net  = QNetwork(state_size, n_actions, lr)
        self.target_net.copy_weights_from(self.online_net)

        self.replay = ReplayBuffer(buffer_capacity)

        self.total_steps   = 0
        self.total_reward  = 0.0
        self.losses: List[float] = []
        self._step_counter = 0
        self.train_every   = 4   # only run backprop every N env steps (speed optimisation)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state: List[float], available_actions: List[int]) -> int:
        if not available_actions:
            return 0
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        return self._best_action(state, available_actions)

    def greedy_action(self, state: List[float], available_actions: List[int]) -> int:
        if not available_actions:
            return 0
        return self._best_action(state, available_actions)

    def _best_action(self, state: List[float], available: List[int]) -> int:
        s = normalise_state(state)
        q = self.online_net.predict(s)
        return max(available, key=lambda a: q[a] if a < len(q) else -999)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        state:          List[float],
        action:         int,
        reward:         float,
        next_state:     List[float],
        done:           bool,
        available_next: List[int],
    ) -> Optional[float]:
        s  = normalise_state(state)
        s2 = normalise_state(next_state)

        self.replay.push(s, action, reward, s2, done)
        self.total_steps  += 1
        self.total_reward += reward
        self._decay_epsilon()

        self._step_counter += 1
        if len(self.replay) < self.batch_size or self._step_counter % self.train_every != 0:
            return None

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        # Compute targets with target network
        q_next, _ = self.target_net.forward(next_states)
        q_next_max = q_next.max(axis=1)
        targets_full = rewards + self.gamma * q_next_max * (1.0 - dones)

        loss = self.online_net.train_step(states, actions, targets_full)
        self.losses.append(loss)

        # Hard update target network
        if self.total_steps % self.target_update == 0:
            self.target_net.copy_weights_from(self.online_net)

        return loss

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "online_W1": self.online_net.W1, "online_b1": self.online_net.b1,
                "online_W2": self.online_net.W2, "online_b2": self.online_net.b2,
                "online_W3": self.online_net.W3, "online_b3": self.online_net.b3,
                "target_W1": self.target_net.W1, "target_b1": self.target_net.b1,
                "target_W2": self.target_net.W2, "target_b2": self.target_net.b2,
                "target_W3": self.target_net.W3, "target_b3": self.target_net.b3,
                "epsilon":       self.epsilon,
                "total_steps":   self.total_steps,
                "total_reward":  self.total_reward,
                "n_actions":     self.n_actions,
                "state_size":    self.state_size,
                "gamma":         self.gamma,
                "batch_size":    self.batch_size,
                "target_update": self.target_update,
            }, f)
        print(f"[DQNAgent] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "DQNAgent":
        with open(path, "rb") as f:
            d = pickle.load(f)
        agent = cls(state_size=d["state_size"], n_actions=d["n_actions"],
                    gamma=d["gamma"], batch_size=d["batch_size"],
                    target_update=d["target_update"])
        for net, prefix in ((agent.online_net,"online"), (agent.target_net,"target")):
            net.W1[:] = d[f"{prefix}_W1"]; net.b1[:] = d[f"{prefix}_b1"]
            net.W2[:] = d[f"{prefix}_W2"]; net.b2[:] = d[f"{prefix}_b2"]
            net.W3[:] = d[f"{prefix}_W3"]; net.b3[:] = d[f"{prefix}_b3"]
        agent.epsilon      = d["epsilon"]
        agent.total_steps  = d["total_steps"]
        agent.total_reward = d["total_reward"]
        print(f"[DQNAgent] Loaded ← {path}  (ε={agent.epsilon:.4f}, steps={agent.total_steps})")
        return agent

    def _decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
