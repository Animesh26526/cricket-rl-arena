"""
environment/cricket_env.py
---------------------------
Context-aware RL environment for cricket batting (v2).

State vector (7 features)
--------------------------
  [0] delivery_idx        integer 0-10
  [1] stumps_idx          0 = Not Touching, 1 = Touching
  [2] wickets_remaining   10 - wickets fallen
  [3] balls_remaining     legal deliveries left
  [4] runs_required       0 in first innings, else target - score
  [5] current_run_rate    runs per over so far (capped at 36)
  [6] required_run_rate   runs per over needed  (0 in first innings)

This richer context allows the agent to learn pressure-aware strategies:
  - Low wickets + high RRR → aggressive shots
  - Comfortable chase     → steady accumulation
  - High wickets remaining → anchor-type play

Actions
-------
Integer index into self.available_shots for the current delivery.

Reward function
---------------
  run scored              → +1.0 per run  (+bonus for 4/6)
  wicket                  → −25.0
  target chased           → +50.0 bonus
"""

import random
from typing import Dict, List, Optional, Tuple

from models.player import Player
from environment.probability_engine import (
    ProbabilityEngine,
    ALL_DELIVERIES,
    DISMISSAL_TYPES,
)
from environment.drs_system import DRSSystem

REWARD_PER_RUN    =  1.0
REWARD_FOUR_BONUS =  1.0
REWARD_SIX_BONUS  =  2.0
REWARD_WICKET     = -25.0
REWARD_WIN        =  50.0

RUN_MAP = {"No Run":0,"1 Run":1,"2 Runs":2,"3 Runs":3,"4 Runs":4,"6 Runs":6}
RR_CAP  = 36.0   # cap run-rate features to avoid huge values


class CricketEnv:
    """
    Single-innings cricket RL environment.

    Parameters
    ----------
    total_balls  : int   Max legal deliveries in the innings.
    match_format : str   "T20", "ODI", or "Test".
    target       : int   Runs needed to win (2nd innings). None for 1st.
    batter_skill : float Batting skill 0.0–1.0 forwarded to ProbabilityEngine.
    verbose      : bool  Print commentary (human play / debug).
    """

    def __init__(
        self,
        total_balls:  int   = 120,
        match_format: str   = "T20",
        target:       Optional[int] = None,
        batter_skill: float = 0.5,
        verbose:      bool  = False,
    ):
        self.total_balls  = total_balls
        self.match_format = match_format
        self.target       = target
        self.batter_skill = batter_skill
        self.verbose      = verbose

        self._engine = ProbabilityEngine()
        self._drs    = DRSSystem()

        self._pp_end = {
            "T20":  max(1, int(total_balls / 6 * 6  / 20)),
            "ODI":  max(1, int(total_balls / 6 * 10 / 50)),
            "Test": 0,
        }

        # runtime (populated by reset)
        self.balls_bowled:    int   = 0
        self.score:           int   = 0
        self.wickets:         int   = 0
        self.powerplay:       bool  = True
        self.delivery:        str   = ""
        self.stumps:          str   = ""
        self.shot:            str   = ""
        self.available_shots: List[str] = []
        self._done:           bool  = False

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def reset(self) -> List[float]:
        self.balls_bowled = 0
        self.score        = 0
        self.wickets      = 0
        self.powerplay    = True
        self._done        = False
        self._sample_delivery()
        return self.get_state()

    def step(self, action: int) -> Tuple[List[float], float, bool, Dict]:
        """Execute one legal delivery. Returns (state, reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode finished — call reset() first.")

        action = max(0, min(action, len(self.available_shots) - 1))
        self.shot = self.available_shots[action]

        result, reward, _ = self._simulate_ball()

        info = {
            "result":       result,
            "runs":         RUN_MAP.get(result, 0),
            "wicket":       result in DISMISSAL_TYPES,
            "shot":         self.shot,
            "delivery":     self.delivery,
            "stumps":       self.stumps,
            "score":        self.score,
            "wickets":      self.wickets,
            "balls_bowled": self.balls_bowled,
        }

        if not self._done:
            self._sample_delivery()

        return self.get_state(), reward, self._done, info

    def get_state(self) -> List[float]:
        """Return context-aware 7-feature state vector."""
        delivery_idx = ALL_DELIVERIES.index(self.delivery) if self.delivery in ALL_DELIVERIES else 0
        stumps_idx   = 1 if self.stumps == "Touching" else 0

        balls_remaining = self.total_balls - self.balls_bowled
        overs_played    = self.balls_bowled / 6.0
        overs_remaining = balls_remaining   / 6.0

        # Run rates
        current_rr  = min(RR_CAP, self.score / overs_played) if overs_played > 0 else 0.0
        runs_required = 0.0
        required_rr   = 0.0
        if self.target is not None:
            runs_required = float(max(0, self.target - self.score))
            if overs_remaining > 0:
                required_rr = min(RR_CAP, runs_required / overs_remaining)

        return [
            float(delivery_idx),
            float(stumps_idx),
            float(10 - self.wickets),
            float(balls_remaining),
            runs_required,
            current_rr,
            required_rr,
        ]

    def get_available_actions(self) -> List[int]:
        return list(range(len(self.available_shots)))

    # ------------------------------------------------------------------
    # Internal simulation
    # ------------------------------------------------------------------

    def _sample_delivery(self) -> None:
        self.delivery        = random.choice(ALL_DELIVERIES)
        self.stumps          = random.choice(["Touching", "Not Touching"])
        self.available_shots = self._engine.get_available_shots(self.delivery, self.stumps)

    def _simulate_ball(self) -> Tuple[str, float, bool]:
        extra = self._engine.sample_extras()
        if extra != "None":
            return self._handle_extra(extra)

        result = self._engine.sample_outcome(
            self.shot, self.delivery, self.stumps,
            self.match_format, self.powerplay, self.batter_skill,
        )
        result = self._apply_rules(result)
        reward = self._apply_result(result)
        return result, reward, False

    def _apply_rules(self, result: str) -> str:
        if self.shot == "Leave":
            return "No Run"
        if result == "L.B.W" and self.stumps == "Not Touching":
            return "Leg Bye"
        while result == "Bowled" and self.stumps == "Not Touching":
            result = self._engine.sample_outcome(
                self.shot, self.delivery, self.stumps,
                self.match_format, self.powerplay, self.batter_skill,
            )
        while result in ("Wide","Wide Four") and self.stumps == "Touching":
            result = self._engine.sample_outcome(
                self.shot, self.delivery, self.stumps,
                self.match_format, self.powerplay, self.batter_skill,
            )
        if result in ("Caught","Caught and Bowled","Edged And Caught Behind"):
            if random.random() < 0.15:
                result = random.choice(["No Run","1 Run","2 Runs"])
        return result

    def _apply_result(self, result: str) -> float:
        if result in DISMISSAL_TYPES:
            self.wickets      += 1
            self.balls_bowled += 1
            self._update_powerplay()
            if self.wickets >= 10:
                self._done = True
            return REWARD_WICKET

        runs = RUN_MAP.get(result, 0)
        self.score        += runs
        self.balls_bowled += 1
        self._update_powerplay()

        reward = runs * REWARD_PER_RUN
        if runs == 4: reward += REWARD_FOUR_BONUS
        if runs == 6: reward += REWARD_SIX_BONUS

        if self.target and self.score >= self.target:
            reward     += REWARD_WIN
            self._done  = True
            if self.verbose: print("Target chased!")

        if self.balls_bowled >= self.total_balls:
            self._done = True

        return reward

    def _handle_extra(self, extra: str) -> Tuple[str, float, bool]:
        if extra == "Wide":        self.score += 1
        elif extra == "Wide Four": self.score += 5
        elif extra == "Leg Bye":
            self.score        += 1
            self.balls_bowled += 1
        elif extra == "No Ball":   self.score += 1
        elif extra == "Run Out":
            self.wickets += 1
            if self.wickets >= 10:
                self._done = True
            return extra, REWARD_WICKET, True
        return extra, 0.0, True

    def _update_powerplay(self) -> None:
        if not self.powerplay:
            return
        if self.balls_bowled // 6 >= self._pp_end.get(self.match_format, 0):
            self.powerplay = False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def state_size(self) -> int:
        return 7

    def render(self) -> None:
        rr = f"{self.score/(self.balls_bowled/6):.2f}" if self.balls_bowled else "0.00"
        print(
            f"Score: {self.score}/{self.wickets}  "
            f"Balls: {self.balls_bowled}/{self.total_balls}  "
            f"RR: {rr}  "
            f"Delivery: {self.delivery:<12}  Stumps: {self.stumps}"
        )
