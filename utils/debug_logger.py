"""
utils/debug_logger.py
---------------------
Runtime logging system for debugging and monitoring game execution.
Uses Python's built-in logging module for structured logging to console and file.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config import LOGGING_CONFIG

# Create logs directory
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging format
LOG_FORMAT = logging.Formatter(
    LOGGING_CONFIG["log_format"], datefmt="%Y-%m-%d %H:%M:%S"
)

# Root logger setup
ROOT_LOGGER = logging.getLogger("cricket_ai")
ROOT_LOGGER.setLevel(getattr(logging, LOGGING_CONFIG["log_level"]))

# File handler
file_handler = logging.FileHandler(LOGGING_CONFIG["log_file"])
file_handler.setLevel(getattr(logging, LOGGING_CONFIG["log_level"]))
file_handler.setFormatter(LOG_FORMAT)
ROOT_LOGGER.addHandler(file_handler)

# Console handler (optional, less verbose)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(LOG_FORMAT)
ROOT_LOGGER.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(f"cricket_ai.{name}")


# Module-specific loggers
match_logger = get_logger("match")
agent_logger = get_logger("agent")
env_logger = get_logger("environment")
