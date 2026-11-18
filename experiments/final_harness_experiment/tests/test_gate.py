from __future__ import annotations

import json
from pathlib import Path

from experiments.final_harness_experiment.baseline_adapter import AUTOSADDLER_COMMIT, probe_official_checkout


ROOT = Path(__file__).resolve().parents[3]


def test_official_checkout_is_pinned_and_importable():
    result = probe_official_checkout(ROOT)
    assert result["ok"] is True
    assert result["commit"] == AUTOSADDLER_COMMIT
    assert result["reproduction_status"] == "exact_or_adapter_reproduction"


def test_preregistration_keeps_heldout_closed_and_registers_exact_baseline():
    protocol = json.loads((ROOT / "experiments/final_harness_experiment/preregistered_protocol.json").read_text(encoding="utf-8"))
    assert protocol["framework"]["name"] == "AutoSaddler V2"
    assert protocol["held_out_opened"] is False
