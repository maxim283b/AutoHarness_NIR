from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.controllers.policies import Fixed
from src.execution.llm import Client, Limits
from src.execution.runner import execute_task
from src.storage.jsonl import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, help="Existing trace with identical model and P1 train rows")
    parser.add_argument("--max-total-seconds", type=float, default=900)
    args = parser.parse_args()
    tasks = read_jsonl(ROOT / "data" / "train.jsonl")
    rows = read_jsonl(args.out) if args.out.exists() else []
    if args.reuse:
        for row in read_jsonl(args.reuse):
            if row.get("split") == "train" and row.get("arm") == "P1" and row["task_id"] not in {r["task_id"] for r in rows}:
                if row.get("model") != os.getenv("OPENAI_MODEL"):
                    parser.error("Reused rows use a different model")
                rows.append(row)
    write_jsonl(args.out, rows)
    start = time.monotonic()
    for task in tasks:
        if task["task_id"] in {r["task_id"] for r in rows}:
            continue
        if time.monotonic() - start >= args.max_total_seconds:
            break
        client = Client.from_env(Limits(10, 4500, 150))
        row = execute_task(task, Fixed("P1"), client)
        row.update({"split": "train", "arm": "P1", "model": client.model})
        rows.append(row)
        write_jsonl(args.out, rows)
        print(f"{len(rows)}/12 task={task['task_id']} success={row['success']} visible_errors={row['errors']} calls={row['calls']} wall={row['wall_s']:.1f}s", flush=True)
    print(json.dumps({"completed": len(rows) == len(tasks), "n": len(rows), "successes": sum(r["success"] for r in rows), "visible_errors": sum(r["errors"] for r in rows), "out": str(args.out)}))


if __name__ == "__main__":
    main()
