# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-?? (Initial Release)

### Added
- **Multiple Match Formats**: T20, ODI, and Test cricket with format-specific rules
- **Reinforcement Learning Agents**:
  - Q-Learning agent with 7-dimensional state space
  - DQN (Deep Q-Network) support
- **Comprehensive Cricket Rules**:
  - Powerplay restrictions
  - DRS (Decision Review System) with 2 reviews per innings
  - Realistic dismissal types (LBW, Bowled, Caught, Stumped)
  - Wides, no-balls, leg byes
  - Run rate and economy tracking
- **Game Modes**:
  - Human vs Human
  - Human vs AI
  - Super Over for tie-breaking
- **Data Tracking**:
  - Match history in JSON format
  - Player statistics (batting & bowling)
  - Training logs and metrics
- **Project Infrastructure**:
  - Well-structured modular architecture
  - Comprehensive README documentation
  - Unit tests for models and helpers
  - CI/CD workflow configuration

### Fixed
- State mismatch: Bowler state now returns exactly 7 values
- Bowler stats: Uses roster players instead of creating new instances
- Error handling for None bowlers
- Separate batter/bowler agent checkpoint support
- Comprehensive logging infrastructure

### Known Limitations
- 7-feature state space may not capture all game complexity
- Tabular Q-learning limited to ~120K discrete states
- Single batter mode (no multi-player partnership dynamics)
- Basic ε-greedy exploration strategy

---

## [Unreleased]

### Planned Features
- [ ] Separate training pipelines for batter & bowler agents
- [ ] GUI interface (PyGame/PyQt)
- [ ] Partnership dynamics (multi-player batting coordination)
- [ ] Field placement strategy
- [ ] Weather & pitch condition effects
- [ ] Advanced state normalization for DQN
- [ ] Multi-format transfer learning
- [ ] Policy gradient algorithms (Actor-Critic)

---

For older changes, please refer to commit history.

---

