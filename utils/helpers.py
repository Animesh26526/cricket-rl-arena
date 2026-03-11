"""
utils/helpers.py
-----------------
Shared utility functions used across the project.
"""

from __future__ import annotations
import random
from typing import List


def get_int_input(prompt: str, lo: int, hi: int) -> int:
    """
    Prompt the user for an integer in [lo, hi] using a while-loop
    (no recursion — safe for unlimited retries).
    """
    while True:
        raw = input(prompt).strip()
        if raw.isdigit():
            value = int(raw)
            if lo <= value <= hi:
                return value
        print(f"  ⚠ Please enter a number between {lo} and {hi}.")


def get_str_input(prompt: str, valid: List[str]) -> str:
    """Prompt for a string that must be one of `valid` (case-insensitive)."""
    valid_upper = [v.upper() for v in valid]
    while True:
        raw = input(prompt).strip().upper()
        if raw in valid_upper:
            return valid[valid_upper.index(raw)]
        print(f"  ⚠ Valid options: {', '.join(valid)}")


def parse_runs(result: str) -> int:
    """Extract the run count from a result string like '4 Runs' → 4."""
    for part in result.split():
        if part.isdigit():
            return int(part)
    return 0


def short_form_result(result: str) -> str:
    """Convert a ball result to its over-summary short form."""
    mapping = {
        "No Run": ".",
        "1 Run": "1",
        "2 Runs": "2",
        "3 Runs": "3",
        "4 Runs": "4",
        "6 Runs": "6",
        "Wide": "WD",
        "Wide Four": "5WD",
        "Leg Bye": "LB",
        "No Ball": "NB",
    }
    if result in mapping:
        return mapping[result]
    # Any dismissal
    return "W"


def weighted_choice(options: List[str], weights: List[float]) -> str:
    """Convenience wrapper around random.choices that returns a single string."""
    return random.choices(options, weights=weights, k=1)[0]


def compute_economy(runs_conceded: int, balls_bowled: int) -> float:
    """Calculate bowling economy rate."""
    if balls_bowled == 0:
        return 0.0
    overs = balls_bowled / 6.0
    return round(runs_conceded / overs, 2)


def compute_strike_rate(runs: int, balls: int) -> float:
    """Calculate batting strike rate."""
    if balls == 0:
        return 0.0
    return round((runs / balls) * 100, 2)


def format_overs(balls: int) -> str:
    """Convert a ball count to 'X.Y overs' format (e.g. 37 → '6.1')."""
    return f"{balls // 6}.{balls % 6}"
