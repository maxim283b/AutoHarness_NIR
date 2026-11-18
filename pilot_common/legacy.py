"""Compatibility adapter for the existing pilot/bandit implementation.

It intentionally delegates control flow, extraction and the shared sandbox
verifier to ``pilot/bandit/patterns.py``. This is useful when a pilot result
must remain comparable with the historical cache; the new schema is only a
serialization boundary around that unchanged implementation.
"""

from __future__ import annotations

import inspect
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BANDIT = HERE / "pilot" / "bandit"
if str(BANDIT) not in sys.path:
    sys.path.insert(0, str(BANDIT))

from patterns import PATTERNS, MemoryBank, run_pattern  # noqa: E402


def existing_harness_source_hash() -> str:
    """Hash the exact legacy patterns source used by the pilot cache."""
    import patterns
    return hashlib.sha256(inspect.getsource(patterns).encode("utf-8")).hexdigest()


def run_existing_pattern(pattern: str, task: dict, model_client, memory_items: list[dict] | None = None) -> dict:
    """Run one unchanged legacy pattern and return its original outcome dict."""
    name = "memory" if pattern == "memory_assisted" else pattern
    if name not in PATTERNS:
        raise ValueError(f"legacy pilot does not define pattern {name!r}")
    return run_pattern(name, task, model_client, MemoryBank(memory_items or []))

