"""Restart the frozen campaign after recoverable worker failures."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).with_name("run.py")
OUT = ROOT / "results/autosaddler_reproduction/improved_v2_campaign"
LOG = OUT / "supervisor.jsonl"
MAX_HOURS = 74


def write(event, **kwargs):
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                                 "event": event, **kwargs}, ensure_ascii=False) + "\n")


def progress():
    result = []
    for label in ("H1", "H2"):
        run = ROOT / "results/autosaddler_reproduction/improved_v2_runs" / (
            "swe-improved-v2-" + label.lower() + "-20261112")
        events = run / "events.jsonl"
        result.append((label, events.stat().st_size if events.exists() else 0,
                       (OUT / label / "complete.json").exists()))
    final = OUT / "final/results"
    result.append(("final", len(list(final.rglob("result.json"))) if final.exists() else 0))
    return result


def main():
    deadline = time.monotonic() + MAX_HOURS * 3600
    same_failure = 0
    previous = None
    while time.monotonic() < deadline:
        before = progress()
        write("worker_start", progress=before)
        with (OUT / "worker.log").open("a", encoding="utf-8") as output:
            completed = subprocess.run([sys.executable, str(SCRIPT), "run"], cwd=ROOT,
                                       stdout=output, stderr=subprocess.STDOUT)
        status_path = OUT / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        after = progress()
        write("worker_exit", code=completed.returncode, progress=after, status=status)
        if completed.returncode == 0 and status.get("status") == "complete":
            return 0
        message = status.get("message", "")
        # Never repeat an ambiguous paid final request or accept a mismatched result.
        if any(term in message for term in ("Interrupted final attempt", "mismatched final result",
                                            "Frozen source changed", "Frozen plan changed", "H0 changed",
                                            "quota", "Quota", "balance", "Balance")):
            write("needs_reconciliation", reason=message)
            return 2
        signature = (completed.returncode, message, str(after))
        same_failure = same_failure + 1 if signature == previous and before == after else 0
        previous = signature
        if same_failure >= 3:
            write("repeated_failure", reason=message)
            return 3
        delay = min(300, 15 * (2 ** same_failure))
        write("automatic_restart_scheduled", delay_seconds=delay, reason=message)
        time.sleep(delay)
    write("time_budget_reached", hours=MAX_HOURS)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
