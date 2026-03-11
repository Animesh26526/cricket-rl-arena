"""
tests/test_models.py
--------------------
Unit tests for Player and Team models.
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.player import Player
from models.team import Team


class TestPlayer(unittest.TestCase):
    """Test cases for the Player class."""

    def setUp(self):
        self.player = Player("Test Player", skill=0.7)

    def test_player_initialization(self):
        """Test player is initialized with correct attributes."""
        self.assertEqual(self.player.name, "Test Player")
        self.assertEqual(self.player.skill, 0.7)
        self.assertEqual(self.player.runs, 0)
        self.assertEqual(self.player.balls, 0)
        self.assertFalse(self.player.dismissed)

    def test_skill_validation(self):
        """Test skill must be between 0.0 and 1.0."""
        with self.assertRaises(ValueError):
            Player("Invalid", skill=1.5)
        with self.assertRaises(ValueError):
            Player("Invalid", skill=-0.1)

    def test_add_runs(self):
        """Test adding runs updates strike rate correctly."""
        self.player.add_runs(4)
        self.assertEqual(self.player.runs, 4)
        self.assertEqual(self.player.balls, 1)
        self.assertEqual(self.player.strike_rate, 400.0)

    def test_bowling_stats(self):
        """Test bowling statistics tracking."""
        self.player.wickets = 2
        self.player.balls_bowled = 36
        self.player.runs_conceded = 35
        self.assertEqual(self.player.balls_bowled, 36)
        self.assertEqual(self.player.runs_conceded, 35)

    def test_player_reset(self):
        """Test player reset clears innings stats but preserves skill."""
        self.player.add_runs(50)
        self.player.wickets = 2
        self.player.reset()
        self.assertEqual(self.player.runs, 0)
        self.assertEqual(self.player.balls, 0)
        self.assertEqual(self.player.wickets, 0)
        self.assertEqual(self.player.skill, 0.7)


class TestTeam(unittest.TestCase):
    """Test cases for the Team class."""

    def setUp(self):
        self.team = Team("Test Team", "Captain")
        for i in range(11):
            self.team.add_player(Player(f"Player{i}"))

    def test_team_initialization(self):
        """Test team is initialized with correct attributes."""
        self.assertEqual(self.team.name, "Test Team")
        self.assertEqual(self.team.captain, "Captain")
        self.assertEqual(self.team.score, 0)
        self.assertEqual(self.team.wickets, 0)

    def test_add_player(self):
        """Test adding players to team."""
        initial_count = len(self.team.players)
        self.team.add_player(Player("New Player"))
        self.assertEqual(len(self.team.players), initial_count + 1)

    def test_add_fast_bowler(self):
        """Test adding fast bowler."""
        self.team.add_fast_bowler("Player0")
        self.assertIn("Player0", self.team.fast_bowlers)

    def test_add_spin_bowler(self):
        """Test adding spin bowler."""
        self.team.add_spin_bowler("Player1")
        self.assertIn("Player1", self.team.spin_bowlers)

    def test_duplicate_bowler_error(self):
        """Test cannot add same bowler to both fast and spin."""
        self.team.add_fast_bowler("Player0")
        with self.assertRaises(ValueError):
            self.team.add_spin_bowler("NonExistentPlayer")

    def test_get_next_batsman(self):
        """Test getting next undismissed batsman."""
        first_bat = self.team.players[0]
        first_bat.dismissed = True
        next_bat = self.team.get_next_batsman()
        self.assertEqual(next_bat.name, "Player1")

    def test_validate_captain_in_roster(self):
        """Test captain validation."""
        self.team.captain = "NonExistentPlayer"
        with self.assertRaises(ValueError):
            self.team.validate_captain()


if __name__ == "__main__":
    unittest.main()
