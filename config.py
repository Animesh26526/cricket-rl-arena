"""
config.py
---------
Centralized configuration file for the Cricket AI Simulator.
Defines constants, game rules, and hyperparameters used across the project.
"""

# Game Rules & Constants
GAME_CONFIG = {
    "players_per_team": 11,
    "max_overs_t20": 20,
    "max_overs_odi": 50,
    "max_overs_test": "unlimited",
    "balls_per_over": 6,
    "powerplay_overs_t20": 6,
    "powerplay_overs_odi": 10,
    "max_reviews_per_innings": 2,
    "catch_drop_probability": 0.15,
}

# Dismissal Types
DISMISSAL_TYPES = {
    "L.B.W",
    "Bowled",
    "Caught",
    "Caught and Bowled",
    "Run Out",
    "Stumped",
    "Edged And Caught Behind",
}

# Run Scoring Map
RUN_MAP = {
    "No Run": 0,
    "1 Run": 1,
    "2 Runs": 2,
    "3 Runs": 3,
    "4 Runs": 4,
    "6 Runs": 6,
}

# Agent Configuration
AGENT_CONFIG = {
    "batter_checkpoint": "training/checkpoints/batter_agent.pkl",
    "bowler_checkpoint": "training/checkpoints/bowler_agent.pkl",
    "default_checkpoint": "training/checkpoints/qlearning_final.pkl",
}

# Q-Learning Agent Hyperparameters
Q_LEARNING_CONFIG = {
    "alpha": 0.1,  # Learning rate
    "gamma": 0.95,  # Discount factor
    "epsilon": 1.0,  # Exploration rate
    "epsilon_min": 0.05,  # Minimum exploration rate
    "epsilon_decay": 0.9995,  # Exploration decay per episode
    "n_deliveries": 11,
    "n_stumps": 2,
    "n_wickets": 11,
    "n_balls_bins": 6,
    "n_rrr_bins": 5,  # required run rate buckets
    "n_crr_bins": 4,  # current run rate buckets
    "n_pressure": 5,  # runs_required buckets
    "n_actions": 14,
}

# DQN Agent Hyperparameters
DQN_CONFIG = {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "memory_size": 10000,
    "batch_size": 32,
    "state_size": 7,  # batter/bowler state dimensions
    "hidden_layer_size": 128,
    "update_frequency": 10,
}

# UI/Display Configuration
UI_CONFIG = {
    "sleep_delivery": 0.5,  # Sleep time before showing delivery
    "sleep_stumps": 0.5,  # Sleep time before showing stumps
    "sleep_result": 0.8,  # Sleep time before showing result
    "sleep_continuation": 1.0,  # Sleep time at various game transitions
    "toss_sleep": 2.0,  # Sleep time for toss animation
}

# Training Configuration
TRAINING_CONFIG = {
    "default_episodes": 10000,
    "default_overs": 20,
    "checkpoint_interval": 1000,
    "save_every_n_episodes": 500,
}

# Logging Configuration
LOGGING_CONFIG = {
    "log_file": "logs/game.log",
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "save_match_history": True,
    "match_history_file": "logs/match_history.json",
}

# State Bounds (for normalization/validation)
STATE_BOUNDS = {
    "delivery_idx": (0, 10),
    "stumps_idx": (0, 1),
    "wickets_remaining": (0, 10),
    "balls_remaining": (0, 120),  # Assuming max 20 overs
    "runs_required": (0, 500),
    "current_rr": (0, 36),
    "required_rr": (0, 36),
}


def get_max_overs(match_format: str) -> int:
    """Get maximum overs for a given match format."""
    format_upper = match_format.upper()
    if format_upper == "T20":
        return GAME_CONFIG["max_overs_t20"]
    elif format_upper == "ODI":
        return GAME_CONFIG["max_overs_odi"]
    elif format_upper == "TEST":
        return float("inf")
    else:
        raise ValueError(f"Unknown match format: {match_format}")


def get_powerplay_overs(match_format: str) -> int:
    """Get powerplay overs for a given match format."""
    format_upper = match_format.upper()
    if format_upper == "T20":
        return GAME_CONFIG["powerplay_overs_t20"]
    elif format_upper == "ODI":
        return GAME_CONFIG["powerplay_overs_odi"]
    else:
        return 0
