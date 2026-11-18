from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.final_harness_experiment.gate_a2_audit import validate_gate_a2_run


def _write_minimal_run(root: Path, *, scenario: str = "sql_harness", target_client: str = "RecordingLMStudioClient", child_id: str = "sha256:child") -> None:
    parent_id = "sha256:parent"
    (root / "candidates" / "parent").mkdir(parents=True)
    (root / "candidates" / child_id.removeprefix("sha256:")).mkdir(parents=True, exist_ok=True)
    (root / "candidates" / "parent" / "candidate.json").write_text('{"x":"before"}', encoding="utf-8")
    (root / "candidates" / child_id.removeprefix("sha256:") / "candidate.json").write_text('{"x":"after"}', encoding="utf-8")
    events = [
        {"event_type": "RunStarted", "payload": {"scenario": {"name": scenario}}},
        {"event_type": "CandidateFinalized", "payload": {"candidate": {"candidate_id": child_id, "parent_ids": [parent_id]}}},
    ]
    for index, kind in enumerate(("evolve", "diagnose_patch", "reflect")):
        sid = f"sha256:s{index}"
        events.extend([
            {"event_type": "SessionStarted", "payload": {"session_id": sid}},
            {"event_type": "ModelUsageObserved", "payload": {"usage": {"input_tokens": 1, "output_tokens": 1}}},
        ])
        session = root / "sessions" / f"s{index}"
        session.mkdir(parents=True)
        (session / "request.json").write_text(json.dumps({"kind": kind}), encoding="utf-8")
    events.append({"event_type": "RunCompleted", "payload": {"development_score": 0.88}})
    (root / "events.jsonl").write_text("\n".join(json.dumps(row) for row in events), encoding="utf-8")
    raw = root / "raw_target_evaluations"
    raw.mkdir()
    row = {
        "target_client": target_client, "target_model": "ministral-3-8b-instruct-2512",
        "target_endpoint": "http://127.0.0.1:1234/v1/chat/completions", "raw_target_calls": [{"raw_model_output": "SELECT 1"}],
        "candidate_id": child_id, "harness_config_hash": "h2", "task_id": "sql-01-00",
        "operation_id": "r:development", "success": True, "tokens": 1000, "tool_calls": 1,
        "reward": 0.88, "prompt_tokens": 900, "completion_tokens": 100, "target_calls": 1,
        "infrastructure_error": None,
    }
    (raw / "row.json").write_text(json.dumps(row), encoding="utf-8")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"scenario": "fake"}, "scenario"),
        ({"target_client": "MockClient"}, "mock target"),
        ({"child_id": "sha256:parent"}, "patch did not change"),
    ],
)
def test_gate_a2_rejects_fake_mock_and_unchanged(tmp_path: Path, override, message) -> None:
    _write_minimal_run(tmp_path, **override)
    with pytest.raises(AssertionError, match=message):
        validate_gate_a2_run(tmp_path)


def test_gate_a2_rejects_unused_patch_unrecomputable_score_and_leak(tmp_path: Path) -> None:
    _write_minimal_run(tmp_path)
    row_path = tmp_path / "raw_target_evaluations" / "row.json"
    row = json.loads(row_path.read_text(encoding="utf-8"))
    row["candidate_id"] = "sha256:other"
    row_path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(AssertionError, match="patched harness was not used"):
        validate_gate_a2_run(tmp_path)

    row["candidate_id"] = "sha256:child"
    row["reward"] = 123.0
    row_path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(AssertionError, match="cannot be recomputed"):
        validate_gate_a2_run(tmp_path)

    row["reward"] = 0.88
    row_path.write_text(json.dumps(row), encoding="utf-8")
    request = next((tmp_path / "sessions").glob("*/request.json"))
    request.write_text('{"forbidden":"reference_sql"}', encoding="utf-8")
    with pytest.raises(AssertionError, match="leaked"):
        validate_gate_a2_run(tmp_path)
