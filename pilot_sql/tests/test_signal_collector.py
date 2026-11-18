import json

from pilot.sql_signal_collect import (
    MAIN_SEED,
    PublicMemoryBank,
    check_smoke,
    check_static_smoke,
    completed_keys,
    leakage_errors,
    make_config,
    select_tasks,
)
from pilot_common import SQLTaskAdapter


def test_selection_is_exact_and_heldout_free():
    selected, memory, manifest = select_tasks()
    ids = [task["task_id"] for _, task in selected]
    assert ids == manifest["first_experiment_ids"]
    assert not set(ids) & set(manifest["splits"]["held-out"])
    assert {task["task_id"] for task in memory} == set(manifest["splits"]["search"])


def test_public_memory_excludes_current_and_private_fields():
    selected, memory, _ = select_tasks()
    task = selected[0][1]
    examples = PublicMemoryBank(memory).retrieve(task["description"])
    assert examples
    assert all(example["task_id"] != task["task_id"] for example in examples)
    assert all("reference_sql" not in example for example in examples)


def test_static_sql_smoke_checks_pass():
    selected, _, _ = select_tasks()
    assert check_static_smoke(SQLTaskAdapter(), selected[0][1]) == []


def test_leakage_guard_finds_markers_and_exact_query():
    selected, _, _ = select_tasks()
    task = selected[0][1]
    base = {
        "task_id": task["task_id"],
        "task_template": task["template_group"],
        "task_features": {},
        "trajectory": [{"kind": "model", "input": "safe", "messages": []}],
    }
    assert leakage_errors(base, [task["reference_sql"]]) == []
    leaked = json.loads(json.dumps(base))
    leaked["trajectory"][0]["input"] = task["reference_sql"]
    assert leakage_errors(leaked, [task["reference_sql"]])


def test_resume_key_includes_config_hash(tmp_path):
    cfg = make_config("direct", MAIN_SEED, "source")
    row = {"task_id": "q", "harness": "direct", "seed": MAIN_SEED,
           "config_hash": cfg.config_hash, "infrastructure_error": None}
    path = tmp_path / "outcomes.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert completed_keys(path) == {("q", "direct", MAIN_SEED, cfg.config_hash)}
