from __future__ import annotations

import pytest

from experiments.final_harness_experiment.models import seed_candidate
from experiments.final_harness_experiment.proposer import DeterministicProposer, validate_safe_feedback


def test_candidate_hash_is_content_addressed_and_lineage_is_preserved():
    seed = seed_candidate()
    child = seed.apply({"stop_rule": seed.components["stop_rule"] + " Be concise."}, "test")
    assert child.candidate_hash != seed.candidate_hash
    assert child.parent_hash == seed.candidate_hash
    assert child.patch == {"stop_rule": child.components["stop_rule"]}


def test_feedback_leakage_is_rejected():
    with pytest.raises(ValueError, match="forbidden"):
        validate_safe_feedback([{"task_id": "x", "reference_sql": "SELECT secret"}])


def test_deterministic_proposer_changes_only_selected_component():
    parent = seed_candidate()
    proposer = DeterministicProposer()
    child = proposer.propose(parent, "repair_rule", [{"success": False, "error_class": "syntax"}], [])
    changed = [name for name in parent.components if parent.components[name] != child.components[name]]
    assert changed == ["repair_rule"]
    assert proposer.calls == 1
