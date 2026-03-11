"""
tests/test_helpers.py
---------------------
Unit tests for utility helper functions.
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.helpers import (
    parse_runs,
    short_form_result,
    format_overs,
    compute_economy,
)


class TestParseRuns(unittest.TestCase):
    """Test cases for parse_runs function."""

    def test_parse_single_run(self):
        self.assertEqual(parse_runs("1 Run"), 1)

    def test_parse_boundary(self):
        self.assertEqual(parse_runs("4 Runs"), 4)
        self.assertEqual(parse_runs("6 Runs"), 6)

    def test_parse_no_run(self):
        self.assertEqual(parse_runs("No Run"), 0)

    def test_parse_multiple_runs(self):
        self.assertEqual(parse_runs("2 Runs"), 2)
        self.assertEqual(parse_runs("3 Runs"), 3)


class TestShortFormResult(unittest.TestCase):
    """Test cases for short_form_result function."""

    def test_short_form_runs(self):
        self.assertEqual(short_form_result("1 Run"), "1")
        self.assertEqual(short_form_result("4 Runs"), "4")

    def test_short_form_dot(self):
        self.assertEqual(short_form_result("No Run"), ".")


class TestFormatOvers(unittest.TestCase):
    """Test cases for format_overs function."""

    def test_format_single_over(self):
        result = format_overs(6)
        self.assertEqual(result, "1.0")

    def test_format_partial_over(self):
        result = format_overs(8)
        self.assertEqual(result, "1.2")

    def test_format_zero_balls(self):
        result = format_overs(0)
        self.assertEqual(result, "0.0")

    def test_format_multiple_overs(self):
        result = format_overs(24)
        self.assertEqual(result, "4.0")


class TestComputeEconomy(unittest.TestCase):
    """Test cases for compute_economy function."""

    def test_economy_full_over(self):
        economy = compute_economy(runs_conceded=12, balls_bowled=6)
        self.assertEqual(economy, 12.0)  # 12 runs in 1 over = 12.0 economy

    def test_economy_multiple_overs(self):
        economy = compute_economy(runs_conceded=30, balls_bowled=36)
        self.assertEqual(economy, 5.0)

    def test_economy_partial_over(self):
        economy = compute_economy(runs_conceded=8, balls_bowled=12)
        self.assertEqual(economy, 4.0)

    def test_economy_zero_runs(self):
        economy = compute_economy(runs_conceded=0, balls_bowled=6)
        self.assertEqual(economy, 0.0)


if __name__ == "__main__":
    unittest.main()
