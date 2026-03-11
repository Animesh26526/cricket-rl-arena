# 🏏 CricketRL Arena - Multi-Agent RL-Based Cricket Game Simulator

> An intelligent cricket match simulator featuring reinforcement learning agents that learn to play strategic batting and bowling decisions.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI/CD](https://img.shields.io/badge/CI/CD-GitHub%20Actions-blue)](https://github.com/Animesh26526/cricket-rl-arena/actions)
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-orange)](CONTRIBUTING.md)

---

## 🎬 Demo

![Training Reward](docs/reward_curve.png)
![Terminal Match](docs/match_output.png)
![Strategy Heatmap](docs/strategy_heatmap.png)

*Training reward graph, terminal match output, and strategy visualization.*

---

## 🎯 Project Highlights

A complete RL-powered cricket simulator demonstrating:

- **Reinforcement Learning**: Q-Learning & DQN agent implementations with 7-dimensional state space
- **Game Development**: Full cricket match simulation with T20, ODI, and Test formats
- **Software Engineering**: Modular architecture, unit testing, CI/CD pipelines
- **AI/ML**: State discretization, reward shaping, agent training pipelines

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.8+ |
| **ML/AI** | NumPy, TensorFlow (DQN), Q-Learning |
| **Testing** | unittest, pytest, pytest-cov |
| **Code Quality** | Black, isort, flake8, mypy |
| **CI/CD** | GitHub Actions |

---

## 🧠 Why Reinforcement Learning?

Cricket is fundamentally a **sequential decision-making problem** under uncertainty. Each delivery presents the batsman and bowler with choices that affect immediate outcomes *and* long-term match results. The game involves:

- **Sequential Decisions**: Every ball is a decision point with cumulative effects
- **Uncertainty**: Weather, pitch conditions, player form create stochastic environments  
- **Long-term Rewards**: Strategic decisions (setting fields, targeting batsmen) pay off over overs, not balls
- **Partial Observability**: Players must act based on incomplete game state information

This project models cricket as a **Markov Decision Process (MDP)** and explores policy learning under stochastic match dynamics. By using Q-Learning and Deep Q-Networks (DQN), we demonstrate how agents can learn optimal batting and bowling strategies through trial and error, maximizing expected returns over complete innings.

---

## 📁 Project Structure

```
cricket-rl-arena/
├── agents/           # RL agents (Q-Learning, DQN implementations)
├── environment/     # Game mechanics & probability engine
├── models/          # Player & Team data models
├── training/        # Agent training & evaluation scripts
├── tests/           # Unit tests with coverage
├── utils/           # Logging & helper utilities
├── main.py          # Interactive game entry point
└── config.py        # Configuration management
```

---

## 🚀 Quick Start

```bash
# Clone & setup
git clone https://github.com/Animesh26526/cricket-rl-arena.git
cd cricket-rl-arena

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
# OR use editable install with dev dependencies
pip install -e ".[dev]"

# Play a match
python main.py

# Train an agent
python training/train_agent.py --episodes 10000
```

---

## 🧠 RL Implementation Details

### State Space (7 Features)
- Delivery type, wickets remaining, balls remaining
- Runs required, current run rate, required run rate

### Q-Learning Agent
- **~116,600 discretized states**
- Learning rate (α): 0.1
- Discount factor (γ): 0.95
- Exploration decay: 0.9995

### Training Results (10,000 episodes)
| Metric | Initial | Final |
|--------|---------|-------|
| Avg Reward | 50 | 280+ |
| Win Rate | ~50% | 65-70% |

---

## 📊 Key Features Implemented

✅ Multiple match formats (T20, ODI, Test)  
✅ DRS (Decision Review System)  
✅ Realistic cricket rules (dismissals, wides, no-balls)  
✅ Human vs Human & Human vs AI modes  
✅ Separate batting/bowling RL agents  
✅ JSON match history logging  
✅ Comprehensive unit tests  

---

## 🧪 Testing

```bash
# Run all tests
python -m unittest discover tests/

# Run with coverage
pytest tests/ -v --cov=.
```

---

## 📈 Development Journey

### Skills Demonstrated
- **Reinforcement Learning**: Q-Learning, DQN, state discretization
- **Game Logic**: Rule-based systems, probability modeling
- **Software Design**: OOP, separation of concerns
- **DevOps**: CI/CD setup, automated testing
- **Documentation**: Clear README, contributing guidelines

### Challenges Overcome
- State representation for complex game scenarios
- Balancing exploration vs exploitation in training
- Implementing comprehensive cricket rules accurately

---

## 🚀 Future Vision

Potential applications include sports analytics, decision optimisation systems, and game AI research.

---

## 🔗 Connect

- **GitHub**: [github.com/Animesh26526](https://github.com/Animesh26526)
- **Project**: [cricket-rl-arena](https://github.com/Animesh26526/cricket-rl-arena)

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

*Built with ❤️ using Python and Reinforcement Learning*

