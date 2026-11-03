"""Thin wrapper around bwe_sb.evaluate so it can be run from the repo root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bwe_sb.evaluate import main

if __name__ == "__main__":
    sys.exit(main())
