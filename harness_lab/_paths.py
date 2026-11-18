"""Path bootstrap so every script can import AutoSaddler and the research repo."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTOSADDLER_SRC = ROOT / "third_party" / "AutoSaddler" / "src"
RESULTS = ROOT / "results" / "harness_lab"

for entry in (AUTOSADDLER_SRC, ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
