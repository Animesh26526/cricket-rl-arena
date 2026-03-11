"""
models/player.py
----------------
Player data class with batting skill attribute.

Skill ranges from 0.0 (worst) to 1.0 (elite).
The ProbabilityEngine scales run weights upward and dismissal weights
downward based on skill, then re-normalises.
"""

from __future__ import annotations


class Player:
    """Represents a cricket player with batting skill and in-game stats."""

    def __init__(self, name: str, skill: float = 0.5):
        """
        Parameters
        ----------
        name  : str   Player display name.
        skill : float 0.0–1.0  (default 0.5 = average).
                      0.9 = elite international batter.
        """
        if not 0.0 <= skill <= 1.0:
            raise ValueError(f"skill must be 0.0–1.0, got {skill}")
        self.name  = name
        self.skill = skill

        # batting stats
        self.runs:    int   = 0
        self.balls:   int   = 0
        self.fours:   int   = 0
        self.sixes:   int   = 0
        self.strike_rate: float = 0.0
        self.dismissed: bool = False
        self.wicket_taking_bowler_name: str = ""
        self.how_out: str = ""

        # bowling stats
        self.wickets:       int = 0
        self.balls_bowled:  int = 0
        self.runs_conceded: int = 0

    def add_runs(self, runs: int) -> None:
        self.runs  += runs
        self.balls += 1
        if self.balls > 0:
            self.strike_rate = round((self.runs / self.balls) * 100, 2)

    def reset(self) -> None:
        """Reset per-innings statistics; skill is preserved."""
        self.runs = self.balls = self.fours = self.sixes = 0
        self.strike_rate = 0.0
        self.dismissed = False
        self.wicket_taking_bowler_name = ""
        self.how_out = ""
        self.wickets = self.balls_bowled = self.runs_conceded = 0

    def __str__(self) -> str:
        marker = "" if not self.dismissed else " "
        return f"{self.name} - {self.runs}({self.balls}){marker}  S/R: {self.strike_rate}"

    def __repr__(self) -> str:
        return f"Player(name={self.name!r}, skill={self.skill})"
