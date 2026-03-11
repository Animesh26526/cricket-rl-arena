#!/usr/bin/env python3
"""
Command-line entry point for the Cricket AI Simulator.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import main

if __name__ == "__main__":
    main()
