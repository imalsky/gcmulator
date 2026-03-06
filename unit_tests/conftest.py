"""Test bootstrap for local source-tree imports."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
MY_SWAMP_SRC = ROOT.parent / "MY_SWAMP" / "src"

for candidate in (SRC_ROOT, MY_SWAMP_SRC):
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
