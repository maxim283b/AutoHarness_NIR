"""Durable request-level token ledger. Unknown usage is not zero usage."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading

ROLES = {"optimizer", "target_search", "target_final_baseline", "target_final_optimized", "judge"}

def synchronized(method):
    def wrapped(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return wrapped


def normalize_opencode(tokens: dict | None) -> dict:
    if not tokens or any(k not in tokens for k in ("input", "output")):
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None, "usage_complete": False}
    cache = tokens.get("cache") or {}
    # OpenCode/AI SDK reports input, cache read and cache write separately.
    # Pinned OpenCode 1.19.5 stores visible output and reasoning separately.
    # Provider/ACP output includes reasoning, but native OpenCode tokens.output
    # has already subtracted it. Recombine those native fields exactly once.
    values = [tokens.get("input", 0), tokens.get("output", 0), tokens.get("reasoning", 0), cache.get("read", 0), cache.get("write", 0)]
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values):
        raise ValueError("Invalid OpenCode token counters")
    uncached, output, reasoning, read, write = values
    input_total = uncached + read + write
    output_total = output + reasoning
    reported = tokens.get("total")
    if reported is not None and reported != input_total + output_total:
        raise ValueError("Native OpenCode total disagrees with its component counters")
    return {"input_tokens": input_total, "output_tokens": output_total,
            "total_tokens": input_total + output_total, "cached_input_tokens": read,
            "cache_write_tokens": write, "reasoning_tokens": reasoning, "visible_output_tokens": output,
            "usage_complete": True, "raw": tokens}


class UsageLedger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS calls (
            request_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, candidate_id TEXT,
            role TEXT NOT NULL, benchmark TEXT, task_id TEXT, phase TEXT,
            payload TEXT NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS reservations (
            id TEXT PRIMARY KEY, amount REAL NOT NULL, status TEXT NOT NULL)""")
        self.connection.commit()

    @synchronized
    def record(self, request_id: str, run_id: str, role: str, usage: dict, **context):
        if role not in ROLES:
            raise ValueError("Unknown usage role")
        payload = {**usage, **context}
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        with self.connection:
            existing = self.connection.execute("SELECT run_id,role,payload FROM calls WHERE request_id=?", (request_id,)).fetchone()
            if existing:
                if existing != (run_id, role, serialized):
                    raise ValueError("Request ID already exists with different usage/context")
                return False
            self.connection.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,?,?)", (
                request_id, run_id, context.get("candidate_id"), role,
                context.get("benchmark"), context.get("task_id"), context.get("phase"), serialized,
            ))
        return True

    @synchronized
    def reserve(self, reservation_id: str, amount: float, limit: float):
        if amount <= 0 or limit <= 0:
            raise ValueError("Positive bounded reservations required")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.connection.execute("SELECT amount FROM reservations WHERE id=?", (reservation_id,)).fetchone()
            if existing:
                raise ValueError("Duplicate reservation; reconcile before retrying")
            committed = self.connection.execute("SELECT COALESCE(SUM(amount),0) FROM reservations WHERE status != 'released'").fetchone()[0]
            if committed + amount > limit:
                raise RuntimeError("Budget exhausted before request")
            self.connection.execute("INSERT INTO reservations VALUES (?,?,'reserved')", (reservation_id, amount))
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    @synchronized
    def settle(self, reservation_id: str, actual: float | None):
        exceeded = False
        with self.connection:
            row = self.connection.execute("SELECT amount,status FROM reservations WHERE id=?", (reservation_id,)).fetchone()
            if row is None or row[1] != "reserved":
                raise ValueError("No active reservation")
            # Unknown costs retain their full reservation, preventing silent overspend.
            if actual is None:
                return
            if actual < 0:
                raise ValueError("Negative cost")
            self.connection.execute("UPDATE reservations SET amount=?,status='settled' WHERE id=?", (actual, reservation_id))
            exceeded = actual > row[0]
        if exceeded:
            raise RuntimeError("Actual cost exceeded request reservation; paid execution must stop")

    @synchronized
    def totals(self, run_id: str, benchmark: str | None = None):
        result = {}
        for role, raw in self.connection.execute("SELECT role,payload FROM calls WHERE run_id=?", (run_id,)):
            row = json.loads(raw)
            if benchmark is not None and row.get("benchmark") != benchmark:
                continue
            bucket = result.setdefault(role, {"calls": 0, "unknown_usage_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            bucket["calls"] += 1
            if not row.get("usage_complete"):
                bucket["unknown_usage_calls"] += 1
            for name in ("input_tokens", "output_tokens", "total_tokens"):
                if row.get(name) is not None:
                    bucket[name] += row[name]
        for bucket in result.values():
            bucket["usage_complete"] = bucket["unknown_usage_calls"] == 0
            if not bucket["usage_complete"]:
                bucket["known_total_tokens"] = bucket["total_tokens"]
                bucket["total_tokens"] = None
        return result

    @synchronized
    def close(self):
        self.connection.close()
