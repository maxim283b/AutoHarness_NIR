"""Minimal direct harness used only for the 5-task smoke test.

It accepts a solver callback, performs one model/solver attempt, then invokes
the shared verifier.  No selection algorithm or training is implemented here.
"""
from .tasks import load_tasks
from .verifier import verify

def run_direct(task, solver, seed=0):
    sql = solver(task)
    result = verify(task, sql)
    return {"seed": seed, "harness": "direct", "task_id": task["task_id"], "input": task["description"], "trajectory": [{"event": "solve", "sql": sql}, {"event": "verify", "result": result}], "result": result, "tool_calls": 1, "tokens": len(sql.split())}

def reference_solver(task):
    # Controlled local smoke fixture only; a real agent would provide the callback.
    return task["reference_sql"]

def smoke_tasks(): return load_tasks()[:5]
