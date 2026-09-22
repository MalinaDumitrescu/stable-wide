#!/usr/bin/env python3
"""CLI entry point for the STABLE-WIDE experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from stablewide.experiment import main

if __name__ == "__main__":
    main()
