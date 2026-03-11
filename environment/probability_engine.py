"""
environment/probability_engine.py
-----------------------------------
Probability tables for the cricket simulator.

v2 changes:
  - sample_outcome() now accepts an optional `skill` parameter (0.0–1.0).
    Higher skill boosts scoring weights and reduces dismissal weights,
    then re-normalises so probabilities always sum to 100.
  - Matrix cache is keyed on (format, powerplay, skill_bucket) to avoid
    recomputing identical distributions for players with similar skill.
"""

import random
from typing import Dict, List, Tuple

Outcome     = Tuple[str, float]
OutcomeList = List[Outcome]

# ---------------------------------------------------------------------------
# Shot / delivery tables
# ---------------------------------------------------------------------------

SHOT_TABLE: Dict[Tuple[str, str], List[str]] = {
    ("Leg Spin",    "Touching"):     ["Straight Drive", "Off Drive", "Cover Drive", "Defensive Shot"],
    ("Leg Spin",    "Not Touching"): ["Straight Drive", "Off Drive", "Cover Drive", "Defensive Shot", "Leave"],
    ("Off Spin",    "Touching"):     ["Straight Drive", "On Drive", "Pull", "Scoop Shot", "Defensive Shot"],
    ("Off Spin",    "Not Touching"): ["Straight Drive", "On Drive", "Pull", "Defensive Shot", "Leave"],
    ("Top Spin",    "Touching"):     ["On Drive", "Pull", "Hook", "Defensive Shot"],
    ("Top Spin",    "Not Touching"): ["Cut", "Square Cut", "Defensive Shot", "Leave"],
    ("Arm Ball",    "Touching"):     ["On Drive", "Sweep", "Leg Glance", "Defensive Shot"],
    ("Arm Ball",    "Not Touching"): ["Off Drive", "Square Cut", "Cut", "Defensive Shot", "Leave"],
    ("Flipper",     "Touching"):     ["Straight Drive", "On Drive", "Leg Glance", "Defensive Shot"],
    ("Flipper",     "Not Touching"): ["Straight Drive", "Off Drive", "Cover Drive", "Defensive Shot", "Leave"],
    ("Straight",    "Touching"):     ["Straight Drive", "On Drive"],
    ("Straight",    "Not Touching"): ["Straight Drive", "Off Drive", "Cover Drive", "Defensive Shot", "Leave"],
    ("In Swing",    "Touching"):     ["Straight Drive", "On Drive", "Scoop Shot", "Defensive Shot"],
    ("In Swing",    "Not Touching"): ["Straight Drive", "Off Drive", "Cover Drive", "Defensive Shot", "Leave"],
    ("Out Swing",   "Touching"):     ["Straight Drive", "Off Drive", "Defensive Shot"],
    ("Out Swing",   "Not Touching"): ["Straight Drive", "Off Drive", "Cover Drive", "Defensive Shot", "Leave"],
    ("Leg Cutter",  "Touching"):     ["Straight Drive", "Off Drive", "Leg Glance", "Defensive Shot"],
    ("Leg Cutter",  "Not Touching"): ["Cut", "Square Cut", "Leave"],
    ("Off Cutter",  "Touching"):     ["Straight Drive", "On Drive", "Pull", "Leg Glance", "Defensive Shot"],
    ("Off Cutter",  "Not Touching"): ["Straight Drive", "Off Drive", "Cover Drive", "Defensive Shot", "Leave"],
    ("Bouncer",     "Touching"):     ["On Drive", "Pull"],
    ("Bouncer",     "Not Touching"): ["Hook", "Leave"],
}

SPIN_DELIVERIES = ["Leg Spin", "Off Spin", "Top Spin", "Arm Ball", "Flipper"]
FAST_DELIVERIES = ["Straight", "In Swing", "Out Swing", "Leg Cutter", "Off Cutter", "Bouncer"]
ALL_DELIVERIES  = SPIN_DELIVERIES + FAST_DELIVERIES

ALL_SHOTS = [
    "Straight Drive", "Off Drive", "Cover Drive", "On Drive",
    "Pull", "Scoop Shot", "Cut", "Square Cut",
    "Defensive Shot", "Leave", "Hook", "Sweep", "Leg Glance",
]

DISMISSAL_TYPES = {
    "L.B.W", "Bowled", "Caught", "Caught and Bowled",
    "Run Out", "Stumped", "Edged And Caught Behind",
}

RUN_OUTCOMES = {"No Run", "1 Run", "2 Runs", "3 Runs", "4 Runs", "6 Runs"}

# ---------------------------------------------------------------------------
# Skill adjustment constants
# ---------------------------------------------------------------------------
# For each unit of skill above 0.5 the run outcomes get a +SKILL_RUN_SCALE
# multiplier bonus, and dismissal outcomes get a -SKILL_DISMISS_SCALE penalty.
SKILL_RUN_SCALE     = 0.6   # multiplier variation across skill range
SKILL_DISMISS_SCALE = 0.7   # dismissal weight reduction at skill=1.0


# ---------------------------------------------------------------------------
# ProbabilityEngine
# ---------------------------------------------------------------------------

class ProbabilityEngine:
    """
    Manages all ball-outcome probability computations.

    Matrix is built lazily and cached per (format, powerplay, skill_bucket)
    so it is NOT recomputed on every delivery.
    """

    def __init__(self):
        self._cache: Dict[Tuple[str, bool, int], Dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_matrix(self, match_format: str, powerplay: bool, skill: float = 0.5) -> Dict:
        """Return the probability matrix, building and caching if needed."""
        skill_bucket = round(skill * 10)          # 0–10 → 11 cache buckets
        key = (match_format, powerplay, skill_bucket)
        if key not in self._cache:
            self._cache[key] = self._build_matrix(match_format, powerplay, skill)
        return self._cache[key]

    def get_available_shots(self, delivery: str, stumps: str) -> List[str]:
        return list(SHOT_TABLE.get((delivery, stumps), ["Defensive Shot", "Leave"]))

    def sample_outcome(
        self,
        shot: str,
        delivery: str,
        stumps: str,
        match_format: str,
        powerplay: bool,
        skill: float = 0.5,
    ) -> str:
        """Sample a single ball outcome, adjusted for batter skill."""
        matrix = self.get_matrix(match_format, powerplay, skill)
        key    = (shot, delivery, stumps)
        outcomes = matrix.get(
            key,
            [("No Run", 60.0), ("1 Run", 20.0), ("Caught", 10.0), ("Run Out", 10.0)],
        )
        labels, weights = zip(*outcomes)
        return random.choices(labels, weights=weights)[0]

    def sample_extras(self) -> str:
        """Sample whether this delivery is an extra."""
        outcomes = self._normalize([
            ("No Ball", 5), ("Wide", 5), ("Leg Bye", 3), ("Run Out", 5),
            ("None", 80),   ("Wide Four", 2),
        ])
        labels, weights = zip(*outcomes)
        return random.choices(labels, weights=weights)[0]

    # ------------------------------------------------------------------
    # Private: matrix construction
    # ------------------------------------------------------------------

    def _build_matrix(self, match_format: str, powerplay: bool, skill: float) -> Dict:
        matrix = {}
        for shot in ALL_SHOTS:
            for delivery in ALL_DELIVERIES:
                for stumps in ("Touching", "Not Touching"):
                    base       = self._base_runs(match_format, powerplay)
                    dismissals = self._get_dismissals(shot, delivery, stumps, match_format)
                    combined   = self._apply_skill(base, dismissals, skill)
                    matrix[(shot, delivery, stumps)] = self._normalize(combined)
        return matrix

    # ------------------------------------------------------------------
    # Skill scaling
    # ------------------------------------------------------------------

    def _apply_skill(
        self,
        base_runs: OutcomeList,
        dismissals: OutcomeList,
        skill: float,
    ) -> OutcomeList:
        """
        Scale run outcomes up and dismissal outcomes down based on skill.

        skill = 0.5 → no change (identity).
        skill = 1.0 → max boost to runs, max reduction to dismissals.
        skill = 0.0 → runs reduced, dismissals boosted.

        The combined list is re-normalised after adjustment so probabilities
        always sum to 1.
        """
        delta = skill - 0.5          # range: -0.5 … +0.5

        adjusted_runs = [
            (label, max(0.1, weight * (1.0 + delta * SKILL_RUN_SCALE)))
            for label, weight in base_runs
        ]
        adjusted_dismissals = [
            (label, max(0.1, weight * (1.0 - delta * SKILL_DISMISS_SCALE)))
            for label, weight in dismissals
        ]
        return adjusted_runs + adjusted_dismissals

    # ------------------------------------------------------------------
    # Base run weights
    # ------------------------------------------------------------------

    def _base_runs(self, match_format: str, powerplay: bool) -> OutcomeList:
        if match_format == "T20":
            if powerplay:
                return [("No Run",15),("1 Run",30),("2 Runs",10),("3 Runs",5),("4 Runs",25),("6 Runs",15)]
            return     [("No Run",25),("1 Run",25),("2 Runs",10),("3 Runs",5),("4 Runs",20),("6 Runs",15)]
        if match_format == "ODI":
            if powerplay:
                return [("No Run",20),("1 Run",30),("2 Runs",15),("3 Runs",5),("4 Runs",20),("6 Runs",10)]
            return     [("No Run",30),("1 Run",30),("2 Runs",10),("3 Runs",5),("4 Runs",15),("6 Runs",5)]
        if match_format == "Test":
            return     [("No Run",30),("1 Run",20),("2 Runs",20),("3 Runs",3),("4 Runs",5),("6 Runs",3)]
        return         [("No Run",25),("1 Run",25),("2 Runs",10),("3 Runs",5),("4 Runs",20),("6 Runs",15)]

    # ------------------------------------------------------------------
    # Dismissal weights
    # ------------------------------------------------------------------

    def _scale(self, base: OutcomeList, match_format: str) -> OutcomeList:
        # reduce dismissal likelihoods more heavily in shorter formats
        scale = {"Test": 0.7, "ODI": 0.85, "T20": 0.7}.get(match_format, 1.0)
        return [(label, max(1, int(w * scale))) for label, w in base]

    def _get_dismissals(self, shot, delivery, stumps, match_format) -> OutcomeList:
        if shot == "Leave":
            if stumps == "Touching":
                return self._scale([("Bowled",60),("L.B.W",20),("Edged And Caught Behind",5)], match_format)
            return []
        if "Defensive" in shot:
            return self._scale([("Caught",10),("L.B.W",5)], match_format)
        if shot == "Scoop Shot" and delivery == "Bouncer":
            return self._scale([("Caught",35),("Stumped",5),("Edged And Caught Behind",10),("Run Out",10)], match_format)
        if shot == "Pull" and delivery == "Bouncer":
            return self._scale([("Caught",20),("Edged And Caught Behind",5),("Run Out",10)], match_format)
        if shot in ("Cut","Square Cut") and delivery in ("Off Cutter","Top Spin","Flipper"):
            return self._scale([("Caught",20),("Edged And Caught Behind",10),("Run Out",10)], match_format)
        if shot == "Sweep Shot" and delivery in ("Top Spin","Leg Spin"):
            return self._scale([("Stumped",15),("Caught",10),("Run Out",10),("Edged And Caught Behind",5)], match_format)
        if shot in ("Straight Drive","On Drive") and delivery in ("Straight","Arm Ball","Leg Cutter"):
            return self._scale([("L.B.W",5),("Caught",5),("Run Out",10)], match_format)
        if shot in ("Off Drive","Cover Drive") and delivery in ("In Swing","Out Swing","Off Cutter"):
            return self._scale([("Caught",10),("Run Out",10),("Edged And Caught Behind",5)], match_format)
        return self._scale(
            [("Caught",15),("L.B.W",10),("Run Out",5),
             ("Edged And Caught Behind",5),("Stumped",5),("Caught and Bowled",5)],
            match_format,
        )

    @staticmethod
    def _normalize(outcomes: OutcomeList) -> OutcomeList:
        total = sum(w for _, w in outcomes)
        if total == 0:
            return outcomes
        return [(label, round((w / total) * 100, 4)) for label, w in outcomes]
