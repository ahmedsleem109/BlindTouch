"""Where things live: the repository root, committed results, and generated output (git-ignored)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "out"          # rendered videos, the experience memory
