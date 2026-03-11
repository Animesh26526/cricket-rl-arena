"""
utils/logger.py
----------------
Match history logger — appends structured JSON records after every innings
or match to  logs/match_history.json.

Log record schema
-----------------
{
  "match_id":     int,
  "timestamp":    "YYYY-MM-DD HH:MM:SS",
  "format":       "T20" | "ODI" | "Test",
  "innings":      int,
  "team_score":   int,
  "wickets":      int,
  "balls_played": int,
  "overs":        float,
  "run_rate":     float,
  "top_scorer":   str,
  "top_scorer_runs": int,
  "top_wicket_taker": str,
  "top_wickets":  int,
  "agent_type":   str,   # "Q-Learning" | "DQN" | "Human" | "Random"
  "extra":        dict   # any additional context (target, won, etc.)
}
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_DIR  = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "match_history.json"


def _ensure_log_file() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        LOG_FILE.write_text("[]")


def _load_records() -> List[Dict]:
    _ensure_log_file()
    try:
        text = LOG_FILE.read_text()
        return json.loads(text) if text.strip() else []
    except json.JSONDecodeError:
        return []


def _save_records(records: List[Dict]) -> None:
    LOG_FILE.write_text(json.dumps(records, indent=2))


def _next_match_id() -> int:
    records = _load_records()
    if not records:
        return 1
    return max(r.get("match_id", 0) for r in records) + 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_innings(
    match_format:  str,
    innings:       int,
    score:         int,
    wickets:       int,
    balls_played:  int,
    players:       Optional[List[Any]] = None,   # List[Player]
    agent_type:    str = "Unknown",
    match_id:      Optional[int] = None,
    extra:         Optional[Dict] = None,
) -> int:
    """
    Append an innings record to the log file.

    Parameters
    ----------
    players : list of Player objects (optional)
        Used to extract top_scorer / top_wicket_taker automatically.

    Returns
    -------
    int  The match_id used for this record.
    """
    if match_id is None:
        match_id = _next_match_id()

    overs    = round(balls_played / 6, 2)
    run_rate = round(score / overs, 2) if overs > 0 else 0.0

    top_scorer        = ""
    top_scorer_runs   = 0
    top_wicket_taker  = ""
    top_wickets_count = 0

    if players:
        by_runs   = max(players, key=lambda p: p.runs,    default=None)
        by_wkts   = max(players, key=lambda p: p.wickets, default=None)
        if by_runs:
            top_scorer      = by_runs.name
            top_scorer_runs = by_runs.runs
        if by_wkts and by_wkts.wickets > 0:
            top_wicket_taker  = by_wkts.name
            top_wickets_count = by_wkts.wickets

    record = {
        "match_id":          match_id,
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "format":            match_format,
        "innings":           innings,
        "team_score":        score,
        "wickets":           wickets,
        "balls_played":      balls_played,
        "overs":             overs,
        "run_rate":          run_rate,
        "top_scorer":        top_scorer,
        "top_scorer_runs":   top_scorer_runs,
        "top_wicket_taker":  top_wicket_taker,
        "top_wickets":       top_wickets_count,
        "agent_type":        agent_type,
        "extra":             extra or {},
    }

    records = _load_records()
    records.append(record)
    _save_records(records)
    return match_id


def log_match_result(
    match_id:     int,
    winner:       str,
    margin:       str,
    match_format: str,
    agent_type:   str = "Unknown",
) -> None:
    """Append a lightweight match-result record (summary only)."""
    records = _load_records()
    records.append({
        "match_id":    match_id,
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "record_type": "match_result",
        "format":      match_format,
        "winner":      winner,
        "margin":      margin,
        "agent_type":  agent_type,
    })
    _save_records(records)


def get_all_records() -> List[Dict]:
    """Return all log records."""
    return _load_records()


def get_records_for_format(match_format: str) -> List[Dict]:
    return [r for r in _load_records() if r.get("format") == match_format]


def summarise_logs() -> Dict:
    """Return aggregate statistics from the full log file."""
    records = [r for r in _load_records() if "team_score" in r]
    if not records:
        return {}
    scores    = [r["team_score"] for r in records]
    wickets   = [r["wickets"]    for r in records]
    run_rates = [r["run_rate"]   for r in records]
    return {
        "total_innings":  len(records),
        "avg_score":      round(sum(scores)    / len(scores),    2),
        "avg_wickets":    round(sum(wickets)   / len(wickets),   2),
        "avg_run_rate":   round(sum(run_rates) / len(run_rates), 2),
        "highest_score":  max(scores),
        "lowest_score":   min(scores),
    }
