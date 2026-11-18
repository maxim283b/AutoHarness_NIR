import sqlite3
import json
from pathlib import Path
from pilot_sql.build_data import build
from pilot_sql.tasks import load_tasks, public_task
from pilot_sql.verifier import verify

def setup_module(): build()

def test_all_tasks_are_partitioned_and_public_view_hides_reference():
    tasks = load_tasks(); assert len(tasks) == 60
    assert len({t["description"] for t in tasks}) == 60
    assert len({t["reference_sql"] for t in tasks}) == 60
    assert all("reference_sql" not in public_task(t) for t in tasks)
    manifest = json.loads((Path(__file__).parents[1] / "splits_manifest.json").read_text(encoding="utf-8"))
    assert {k: len(v) for k, v in manifest["splits"].items()} == {"search": 15, "validation": 15, "online": 15, "held-out": 15}
    assert len(set(sum(manifest["splits"].values(), []))) == 60
    by_id = {t["task_id"]: t for t in tasks}
    expected_types = {t["features"]["type"] for t in tasks}
    assert all({by_id[i]["features"]["type"] for i in ids} == expected_types
               for ids in manifest["splits"].values())
    group_to_split = {}
    for split, ids in manifest["splits"].items():
        for task_id in ids:
            group = by_id[task_id]["template_group"]
            assert group_to_split.setdefault(group, split) == split

def test_reference_queries_pass_and_wrong_result_is_structured():
    task = tasks = load_tasks()[0]
    assert verify(task, task["reference_sql"])["success"]
    bad = verify(task, "SELECT name FROM customers")
    assert not bad["success"] and bad["error_code"] == "wrong_result"
    assert "SELECT" not in bad["error"]

def test_mutations_and_multiple_statements_are_rejected():
    task = load_tasks()[0]
    for sql, code in [("DELETE FROM customers", "unsafe_query"), ("DROP TABLE customers", "unsafe_query"), ("SELECT 1; SELECT 2", "multiple_statements"), ("UPDATE customers SET name='x'", "unsafe_query")]:
        result = verify(task, sql); assert result["error_code"] == code

def test_order_sensitive_task_rejects_wrong_order():
    task = next(t for t in load_tasks() if t["order_significant"])
    sql = "SELECT customer_id, name FROM customers ORDER BY customer_id DESC"
    assert verify(task, sql)["error_code"] == "wrong_result"

def test_timeout_is_reported_without_leaking_reference():
    task = load_tasks()[0]
    result = verify(task, "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT n FROM x", timeout_s=0.001)
    assert result["error_code"] in {"timeout", "unsafe_query", "sql_error"}
    assert "reference_sql" not in result
