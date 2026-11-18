"""Independent Gate A2 evidence validation from official events and raw target rows."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


FORBIDDEN_PROPOSER_FIELD = "reference_sql"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


def validate_gate_a2_run(run_dir: Path) -> dict[str, Any]:
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    started = next(event for event in events if event["event_type"] == "RunStarted")
    scenario = started["payload"]["scenario"]["name"]
    assert scenario not in {"fake", "mock"}, "Gate A2 scenario must be real"
    assert scenario == "sql_harness"

    raw_paths = sorted((run_dir / "raw_target_evaluations").glob("*.json"))
    assert raw_paths, "real target rows are missing"
    rows = [_json(path) for path in raw_paths]
    assert all(row["target_client"] == "RecordingLMStudioClient" for row in rows), "mock target client detected"
    assert all(row["target_model"] == "ministral-3-8b-instruct-2512" for row in rows)
    assert any(row["raw_target_calls"] for row in rows), "raw LM Studio outputs are missing"

    finalized = next(
        event
        for event in events
        if event["event_type"] == "CandidateFinalized" and event["payload"].get("kind") != "composition"
    )
    child = finalized["payload"]["candidate"]
    child_id = child["candidate_id"]
    parent_id = child["parent_ids"][0]
    assert child_id != parent_id, "patch did not change harness hash"
    parent_config = _json(run_dir / "candidates" / parent_id.removeprefix("sha256:") / "candidate.json")
    child_config = _json(run_dir / "candidates" / child_id.removeprefix("sha256:") / "candidate.json")
    assert parent_config != child_config, "public harness configs are identical"
    child_rows = [row for row in rows if row["candidate_id"] == child_id]
    assert child_rows, "patched harness was not used in a target evaluation"
    assert all(row["harness_config_hash"] == child_rows[0]["harness_config_hash"] for row in child_rows)

    valid_rows = [row for row in rows if row["reward"] is not None]
    for row in valid_rows:
        recomputed = float(bool(row["success"])) - 0.10 * int(row["tokens"]) / 1000.0 - 0.02 * int(row["tool_calls"])
        assert math.isclose(recomputed, float(row["reward"]), abs_tol=1e-12), "raw-row reward cannot be recomputed"
    assert all(row.get("infrastructure_error") is None or row["success"] is None for row in rows)

    proposer_requests = list((run_dir / "sessions").glob("*/request.json"))
    assert len(proposer_requests) == 3, "official three-session loop did not complete exactly once"
    for path in proposer_requests:
        assert FORBIDDEN_PROPOSER_FIELD not in path.read_text(encoding="utf-8").lower(), "gold SQL field leaked to proposer"

    session_started = [event for event in events if event["event_type"] == "SessionStarted"]
    session_completed = [event for event in events if event["event_type"] == "SessionCompleted"]
    usages = [event["payload"]["usage"] for event in events if event["event_type"] == "ModelUsageObserved"]
    completed = [event for event in events if event["event_type"] == "RunCompleted"]
    assert len(session_started) == 3 and completed, "official AutoSaddler did not complete"
    target_input = sum(int(row["prompt_tokens"]) for row in rows)
    target_output = sum(int(row["completion_tokens"]) for row in rows)
    development_rows = [row for row in rows if ":development" in row["operation_id"]]
    independent_development_score = (
        sum(float(row["reward"]) for row in development_rows if row["reward"] is not None)
        / sum(row["reward"] is not None for row in development_rows)
    )
    diagnosis = next(
        event for event in session_completed if event["payload"]["logical_operation_id"].endswith(":diagnose-patch")
    )
    acceptance = next(event for event in events if event["event_type"] == "AcceptanceDecided")
    parent_rows = [row for row in rows if row["candidate_id"] == parent_id]
    return {
        "schema_version": "gate-a2-audit/v1",
        "gate": "real_target_benchmark_integration",
        "passed": True,
        "scenario": scenario,
        "sql_task_ids": sorted({row["task_id"] for row in rows}),
        "target_model": rows[0]["target_model"],
        "target_endpoint": rows[0]["target_endpoint"],
        "parent_candidate_id": parent_id,
        "child_candidate_id": child_id,
        "selected_candidate_id": completed[0]["payload"]["selected_candidate_id"],
        "proposed_patch": diagnosis["payload"]["result"]["structured_output"],
        "patch_acceptance": acceptance["payload"]["verdict"],
        "parent_harness_config": parent_config,
        "child_harness_config": child_config,
        "parent_harness_config_hash": parent_rows[0]["harness_config_hash"],
        "child_harness_config_hash": child_rows[0]["harness_config_hash"],
        "independently_recomputed_development_score": independent_development_score,
        "official_sessions": [
            {
                "session_id": event["payload"]["session_id"],
                "session_type": event["payload"]["logical_operation_id"].split(":")[-1],
                "stage": event["payload"]["stage"],
            }
            for event in session_started
        ],
        "proposer_calls": len(usages),
        "proposer_input_tokens": sum(int(usage["input_tokens"]) for usage in usages),
        "proposer_output_tokens": sum(int(usage["output_tokens"]) for usage in usages),
        "target_calls": sum(int(row["target_calls"]) for row in rows),
        "target_input_tokens": target_input,
        "target_output_tokens": target_output,
        "infrastructure_errors": sum(row["infrastructure_error"] is not None for row in rows),
        "run_completed": completed[0],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = validate_gate_a2_run(args.run_dir)
    output = args.run_dir / "gate_a2_integrity_report.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
