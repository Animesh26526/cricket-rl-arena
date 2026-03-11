"""
environment/multi_agent_env.py
--------------------------------
Multi-agent RL environment for AI-vs-AI cricket.

The batter agent controls shot selection.
The bowler agent controls delivery type.

This is a zero-sum competitive environment:
  batter_reward  = standard CricketEnv reward (runs − wickets)
  bowler_reward  = −batter_reward  (inverse)

Interface
---------
    env = MultiAgentCricketEnv(total_balls=120, match_format="T20")

    batter_state, bowler_state = env.reset()

    while not done:
        ba = batter_agent.choose_action(batter_state, env.batter_actions())
        bwa = bowler_agent.choose_action(bowler_state, env.bowler_actions())
        (b_state, bw_state), (b_reward, bw_reward), done, info = env.step(ba, bwa)

Batter state  (7 features) — same as CricketEnv.get_state()
Bowler state  (6 features):
  [wickets_taken, balls_bowled_bucket, runs_conceded_bucket,
   current_economy, innings_phase, pressure_flag]
"""

import random
from typing import Dict, List, Optional, Tuple

from environment.probability_engine import (
    ProbabilityEngine,
    ALL_DELIVERIES,
    FAST_DELIVERIES,
    SPIN_DELIVERIES,
    DISMISSAL_TYPES,
)
from environment.drs_system import DRSSystem

RUN_MAP = {"No Run":0,"1 Run":1,"2 Runs":2,"3 Runs":3,"4 Runs":4,"6 Runs":6}

REWARD_PER_RUN    =  1.0
REWARD_FOUR_BONUS =  1.0
REWARD_SIX_BONUS  =  2.0
REWARD_WICKET_BAT = -25.0
REWARD_WIN        =  50.0

BOWLER_WICKET_BONUS    =  15.0
BOWLER_DOT_BALL_BONUS  =   0.5
BOWLER_ECONOMY_PENALTY = -0.3   # per run conceded above 8/over threshold


class MultiAgentCricketEnv:
    """
    Competitive AI-vs-AI cricket environment.

    Parameters
    ----------
    total_balls  : int
    match_format : str   "T20", "ODI", or "Test"
    target       : int | None
    batter_skill : float 0.0–1.0
    """

    BOWLER_STATE_SIZE = 6
    BATTER_STATE_SIZE = 7

    def __init__(
        self,
        total_balls:  int   = 120,
        match_format: str   = "T20",
        target:       Optional[int] = None,
        batter_skill: float = 0.5,
    ):
        self.total_balls  = total_balls
        self.match_format = match_format
        self.target       = target
        self.batter_skill = batter_skill

        self._engine = ProbabilityEngine()
        self._drs    = DRSSystem()

        self._pp_end = {
            "T20":  max(1, int(total_balls / 6 * 6  / 20)),
            "ODI":  max(1, int(total_balls / 6 * 10 / 50)),
            "Test": 0,
        }

        # runtime state
        self.balls_bowled:    int   = 0
        self.score:           int   = 0
        self.wickets:         int   = 0
        self.powerplay:       bool  = True
        self.delivery:        str   = ""
        self.stumps:          str   = ""
        self.available_shots: List[str] = []
        self._done:           bool  = False

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[List[float], List[float]]:
        """Reset and return (batter_state, bowler_state)."""
        self.balls_bowled = 0
        self.score        = 0
        self.wickets      = 0
        self.powerplay    = True
        self._done        = False
        self.stumps       = random.choice(["Touching", "Not Touching"])
        self.delivery     = random.choice(ALL_DELIVERIES)
        self.available_shots = self._engine.get_available_shots(self.delivery, self.stumps)
        return self.get_batter_state(), self.get_bowler_state()

    def step(
        self,
        batter_action: int,
        bowler_action: int,
    ) -> Tuple[Tuple[List[float], List[float]], Tuple[float, float], bool, Dict]:
        """
        Execute one delivery with both agents acting.

        Parameters
        ----------
        batter_action : int  index into available_shots
        bowler_action : int  index into ALL_DELIVERIES

        Returns
        -------
        (batter_state, bowler_state), (batter_reward, bowler_reward), done, info
        """
        if self._done:
            raise RuntimeError("Episode finished — call reset().")

        # Bowler chooses delivery
        bowler_action = max(0, min(bowler_action, len(ALL_DELIVERIES) - 1))
        self.delivery = ALL_DELIVERIES[bowler_action]
        self.stumps   = random.choice(["Touching", "Not Touching"])

        # Update available shots for batter
        self.available_shots = self._engine.get_available_shots(self.delivery, self.stumps)
        batter_action = max(0, min(batter_action, len(self.available_shots) - 1))
        shot = self.available_shots[batter_action]

        result, batter_reward = self._simulate(shot)
        bowler_reward = -batter_reward   # zero-sum

        # Prepare next state
        if not self._done:
            self.delivery = random.choice(ALL_DELIVERIES)
            self.stumps   = random.choice(["Touching", "Not Touching"])
            self.available_shots = self._engine.get_available_shots(self.delivery, self.stumps)

        info = {
            "result":   result,
            "runs":     RUN_MAP.get(result, 0),
            "wicket":   result in DISMISSAL_TYPES,
            "shot":     shot,
            "delivery": self.delivery,
        }
        return (
            (self.get_batter_state(), self.get_bowler_state()),
            (batter_reward, bowler_reward),
            self._done,
            info,
        )

    # ------------------------------------------------------------------
    # State representations
    # ------------------------------------------------------------------

    def get_batter_state(self) -> List[float]:
        """7-feature batter state (identical to CricketEnv)."""
        from environment.probability_engine import ALL_DELIVERIES as AD
        delivery_idx    = AD.index(self.delivery) if self.delivery in AD else 0
        stumps_idx      = 1 if self.stumps == "Touching" else 0
        balls_remaining = self.total_balls - self.balls_bowled
        overs_played    = self.balls_bowled / 6.0
        overs_remaining = balls_remaining   / 6.0
        crr             = min(36.0, self.score / overs_played) if overs_played > 0 else 0.0
        runs_req        = max(0, self.target - self.score) if self.target else 0
        rrr             = min(36.0, runs_req / overs_remaining) if (overs_remaining > 0 and self.target) else 0.0
        return [float(delivery_idx), float(stumps_idx), float(10 - self.wickets),
                float(balls_remaining), float(runs_req), crr, rrr]

    def get_bowler_state(self) -> List[float]:
        """
        6-feature bowler state.
        [wickets_taken, balls_bowled_bucket, runs_conceded_bucket,
         current_economy, innings_phase, pressure_flag]
        """
        economy    = min(36.0, self.score / (self.balls_bowled / 6.0)) if self.balls_bowled >= 6 else 0.0
        bl_bucket  = min(5, self.balls_bowled // 20)
        sc_bucket  = min(5, self.score // 30)
        phase      = min(3, self.balls_bowled // (max(1, self.total_balls) // 4))
        pressure   = 1.0 if (self.target and self.score > self.target * 0.7) else 0.0
        return [
            float(self.wickets),
            float(bl_bucket),
            float(sc_bucket),
            float(economy),
            float(phase),
            float(pressure),
        ]

    def batter_actions(self) -> List[int]:
        return list(range(len(self.available_shots)))

    def bowler_actions(self) -> List[int]:
        return list(range(len(ALL_DELIVERIES)))

    # ------------------------------------------------------------------
    # Internal simulation
    # ------------------------------------------------------------------

    def _simulate(self, shot: str) -> Tuple[str, float]:
        extra = self._engine.sample_extras()
        if extra != "None":
            return self._handle_extra(extra)

        result = self._engine.sample_outcome(
            shot, self.delivery, self.stumps,
            self.match_format, self.powerplay, self.batter_skill,
        )
        result = self._apply_rules(shot, result)
        reward = self._apply_result(result)
        return result, reward

    def _apply_rules(self, shot: str, result: str) -> str:
        if shot == "Leave":
            return "No Run"
        if result == "L.B.W" and self.stumps == "Not Touching":
            return "Leg Bye"
        while result == "Bowled" and self.stumps == "Not Touching":
            result = self._engine.sample_outcome(
                shot, self.delivery, self.stumps,
                self.match_format, self.powerplay, self.batter_skill,
            )
        while result in ("Wide", "Wide Four") and self.stumps == "Touching":
            result = self._engine.sample_outcome(
                shot, self.delivery, self.stumps,
                self.match_format, self.powerplay, self.batter_skill,
            )
        if result in ("Caught", "Caught and Bowled", "Edged And Caught Behind"):
            if random.random() < 0.15:
                result = random.choice(["No Run", "1 Run", "2 Runs"])
        return result

    def _apply_result(self, result: str) -> float:
        if result in DISMISSAL_TYPES:
            self.wickets      += 1
            self.balls_bowled += 1
            self._update_pp()
            if self.wickets >= 10: self._done = True
            return REWARD_WICKET_BAT

        runs = RUN_MAP.get(result, 0)
        self.score        += runs
        self.balls_bowled += 1
        self._update_pp()

        reward = runs * REWARD_PER_RUN
        if runs == 4: reward += REWARD_FOUR_BONUS
        if runs == 6: reward += REWARD_SIX_BONUS
        if self.target and self.score >= self.target:
            reward    += REWARD_WIN
            self._done = True
        if self.balls_bowled >= self.total_balls:
            self._done = True
        return reward

    def _handle_extra(self, extra: str) -> Tuple[str, float]:
        if extra == "Wide":        self.score += 1
        elif extra == "Wide Four": self.score += 5
        elif extra == "Leg Bye":
            self.score += 1; self.balls_bowled += 1
        elif extra == "No Ball":   self.score += 1
        elif extra == "Run Out":
            self.wickets += 1
            if self.wickets >= 10: self._done = True
            return extra, REWARD_WICKET_BAT
        return extra, 0.0

    def _update_pp(self):
        if self.powerplay and self.balls_bowled // 6 >= self._pp_end.get(self.match_format, 0):
            self.powerplay = False
