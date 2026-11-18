"""Read-only, timeout-bounded verifier for the local SQL benchmark."""
from __future__ import annotations
import json, shutil, sqlite3, tempfile, time
from pathlib import Path
from typing import Any
from .tasks import DB_PATH, load_tasks, rows

class VerificationError(Exception):
    pass

def _init_db(path: Path) -> None:
    src = DB_PATH
    if src.exists():
        shutil.copy2(src, path)
        return
    con = sqlite3.connect(path)
    con.executescript("""CREATE TABLE customers(customer_id INTEGER PRIMARY KEY, name TEXT, region TEXT, signup_date TEXT); CREATE TABLE products(product_id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL); CREATE TABLE orders(order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT); CREATE TABLE order_items(order_id INTEGER, product_id INTEGER, quantity INTEGER, unit_price REAL); CREATE TABLE employees(employee_id INTEGER PRIMARY KEY, name TEXT, department TEXT, manager_id INTEGER, hire_date TEXT, salary REAL); CREATE TABLE events(event_id INTEGER PRIMARY KEY, customer_id INTEGER, event_date TEXT, event_type TEXT, value REAL);""")
    for table, values in rows().items():
        marks = ",".join("?" for _ in values[0])
        con.executemany(f"INSERT INTO {table} VALUES ({marks})", values)
    con.commit(); con.close()

def _authorizer(action, arg1, arg2, dbname, source):
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_TRANSACTION}
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

def _normalise(rows_):
    return [tuple((x.isoformat() if hasattr(x, "isoformat") else x) for x in row) for row in rows_]

def verify(task: dict[str, Any], candidate_sql: str, timeout_s: float = 1.0) -> dict[str, Any]:
    started = time.perf_counter()
    base = {"task_id": task["task_id"], "success": False, "error": None, "error_code": None, "elapsed_s": 0.0, "row_count": None}
    if not isinstance(candidate_sql, str) or not candidate_sql.strip():
        base.update(error_code="empty_query", error="Query is empty"); return base
    sql = candidate_sql.strip()
    if ";" in sql.rstrip(";"):
        base.update(error_code="multiple_statements", error="Exactly one read-only statement is required"); return base
    if sql.rstrip().endswith(";"): sql = sql.rstrip()[:-1]
    with tempfile.TemporaryDirectory(prefix="pilot_sql_verify_") as td:
        path = Path(td) / "copy.sqlite"; _init_db(path)
        con = sqlite3.connect(path); con.set_authorizer(_authorizer)
        deadline = time.perf_counter() + timeout_s
        con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 1000)
        try:
            if not sql.lstrip().upper().startswith(("SELECT", "WITH")):
                raise VerificationError("Only SELECT or WITH is allowed")
            got = _normalise(con.execute(sql).fetchall())
            expected = _normalise(con.execute(task["reference_sql"]).fetchall())
            if not task["order_significant"]:
                got, expected = sorted(got, key=repr), sorted(expected, key=repr)
            base["row_count"] = len(got)
            if got != expected:
                base.update(error_code="wrong_result", error="Result does not match the expected result")
            else: base["success"] = True
        except sqlite3.OperationalError as e:
            msg = str(e).lower(); code = "timeout" if "interrupt" in msg else "sql_error"
            base.update(error_code=code, error="Query timed out" if code == "timeout" else "Invalid or unsupported SQL")
        except (sqlite3.DatabaseError, VerificationError):
            base.update(error_code="unsafe_query", error="Query is not an allowed read-only query")
        finally: con.close()
    base["elapsed_s"] = time.perf_counter() - started
    return base

def task_by_id(task_id: str):
    return next((t for t in load_tasks() if t["task_id"] == task_id), None)
