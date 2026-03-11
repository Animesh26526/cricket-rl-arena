# Contributing to RL-Based Cricket Game Simulator

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## 🚀 Quick Start

```bash
# Fork the repository
# Clone your fork
git clone https://github.com/Animesh26526/cricket-rl-arena.git
cd cricket-rl-arena

# Create a virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m unittest discover tests/
```

## 🎯 Ways to Contribute

### Bug Reports
- Use GitHub Issues to report bugs
- Include:
  - Clear description of the issue
  - Steps to reproduce
  - Expected vs actual behavior
  - Python version and environment

### Feature Requests
- Open an Issue with `feature-request` label
- Describe the feature and its use case
- Discuss implementation approach

### Code Contributions
1. **Fork** the repository
2. Create a **feature branch**: `git checkout -b feature/your-feature`
3. Make your **changes**
4. Add **tests** for new functionality
5. Ensure **code quality** (run linting)
6. **Commit** with descriptive messages
7. **Push** to your fork
8. Create a **Pull Request**

## 📝 Coding Standards

### Python Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use **Black** for formatting (line-length: 88)
- Use **isort** for imports
- Add type hints where possible

### Code Quality Tools
```bash
# Format code
black .
isort .

# Lint code
flake8 .

# Type checking
mypy .
```

### Testing Requirements
- All new code must include tests
- Run existing tests before submitting PR:
  ```bash
  python -m unittest discover tests/
  ```
- Target for new code: >80% coverage

## 🏗️ Project Structure

```
cricket-rl-arena/
├── agents/           # RL agent implementations
├── environment/      # Game environment
├── models/          # Data models
├── training/        # Training scripts
├── utils/           # Helper utilities
├── tests/           # Test suite
└── main.py          # Entry point
```

## 🔧 Development Workflow

### Running the Game
```bash
python main.py
```

### Training Agents
```bash
# Train Q-learning agent
python training/train_agent.py --episodes 10000

# Evaluate trained agent
python training/evaluate_agent.py
```

### Running Tests
```bash
# All tests
python -m unittest discover tests/

# Specific test file
python -m unittest tests.test_models

# With pytest
pytest tests/ -v
```

## 📋 Pull Request Guidelines

- PRs should be against the `main` branch
- Include a clear description of changes
- Link related issues
- Ensure all tests pass
- Update documentation if needed

## ❓ Getting Help

- Open a [GitHub Issue](https://github.com/Animesh26526/cricket-rl-arena/issues)
- Check existing issues before creating new ones

---

**Thank you for contributing! 🎉**

