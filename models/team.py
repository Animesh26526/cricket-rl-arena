"""
models/team.py
--------------
Defines the Team class, which manages players, bowlers, and innings state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from models.player import Player


class Team:
    """Represents a cricket team."""

    def __init__(self, name: str, captain: str):
        self.name = name
        self.captain = captain

        # Roster
        self.players: List[Player] = []
        self.fast_bowlers: List[str] = []
        self.spin_bowlers: List[str] = []

        # Innings state
        self.score: int = 0
        self.wickets: int = 0
        self.run_rate: float = 0.0
        self.reviews_left: int = 2

        # Session / day tracking (Test matches)
        self.session: int = 1
        self.day: int = 1

        # Super over
        self.super_over: bool = False
        self.super_over_players: List[Player] = []

        # Bowling state
        self.current_bowler: Optional[Player] = None
        self.current_bowler_name: str = ""

    # ------------------------------------------------------------------
    # Roster management
    # ------------------------------------------------------------------

    def add_player(self, player: "Player") -> None:
        self.players.append(player)

    def add_fast_bowler(self, name: str) -> None:
        """Add a fast bowler by name (must exist in players list)."""
        if not any(p.name == name for p in self.players):
            raise ValueError(f"Fast bowler '{name}' not found in players list")
        if name not in self.fast_bowlers:
            self.fast_bowlers.append(name)

    def add_spin_bowler(self, name: str) -> None:
        """Add a spin bowler by name (must exist in players list)."""
        if not any(p.name == name for p in self.players):
            raise ValueError(f"Spin bowler '{name}' not found in players list")
        if name not in self.spin_bowlers:
            self.spin_bowlers.append(name)

    def validate_captain(self) -> None:
        """Ensure captain is in the players list."""
        if not any(p.name == self.captain for p in self.players):
            raise ValueError(f"Captain '{self.captain}' not found in players list")

    # ------------------------------------------------------------------
    # Batting helpers
    # ------------------------------------------------------------------

    def get_next_batsman(self) -> Optional["Player"]:
        """Return the next undismissed player who hasn't batted yet."""
        pool = self.super_over_players if self.super_over else self.players
        for player in pool:
            if not player.dismissed and player.balls == 0:
                return player
        return None

    # ------------------------------------------------------------------
    # Scorecard helpers
    # ------------------------------------------------------------------

    @staticmethod
    def short_form_dismissal(how_out: str) -> str:
        """Convert a dismissal description to its short scorecard notation."""
        mapping = {
            "Caught and Bowled": "c & b",
            "L.B.W": "lbw",
            "Stumped": "st",
        }
        if how_out in mapping:
            return mapping[how_out]
        if how_out in ("Caught", "Edged And Caught Behind"):
            return "c"
        return how_out

    def print_scorecard(self) -> None:
        print(f"\n--- {self.name} Scorecard ---\n")
        for p in self.players:
            if p.balls == 0 and not p.dismissed:
                continue
            status = f" - {p.runs}({p.balls})"
            if p.dismissed:
                short = self.short_form_dismissal(p.how_out)
                if p.how_out == "Bowled":
                    status += (
                        f"  4s: {p.fours}  6s: {p.sixes}"
                        f"  S/R: {p.strike_rate}"
                        f"  b {p.wicket_taking_bowler_name}"
                    )
                else:
                    status += (
                        f"  4s: {p.fours}  6s: {p.sixes}"
                        f"  S/R: {p.strike_rate}"
                        f"  ({short}) b {p.wicket_taking_bowler_name}"
                    )
            else:
                status += f"*  4s: {p.fours}  6s: {p.sixes}  S/R: {p.strike_rate}"
            print(f"{p.name}{status}")

    def print_bowler_scorecard(self) -> None:
        print("\nBowler Scorecard:")
        for bowler in self.players:
            if bowler.balls_bowled == 0:
                continue
            overs = bowler.balls_bowled // 6 + (bowler.balls_bowled % 6) / 10
            if bowler.balls_bowled >= 6:
                economy = round(bowler.runs_conceded / (bowler.balls_bowled / 6), 2)
            else:
                economy = round(bowler.runs_conceded * (6 / bowler.balls_bowled), 2)
            print(
                f"{bowler.name}: "
                f"Overs: {overs:.1f}  "
                f"Runs: {bowler.runs_conceded}  "
                f"Wickets: {bowler.wickets}  "
                f"Economy: {economy}"
            )

    def __repr__(self) -> str:
        return f"Team(name={self.name!r})"
