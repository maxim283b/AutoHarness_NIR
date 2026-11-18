from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.controllers.policies import Fixed
from src.execution.llm import Client, Limits
from src.execution.runner import execute_task
from src.storage.jsonl import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--protocol", choices=["P1", "P2", "P3", "P4"], default="P1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=90)
    parser.add_argument("--max-calls", type=int, default=3)
    args = parser.parse_args()
    task = read_jsonl(ROOT / "data" / f"{args.split}.jsonl")[args.index]
    client = Client.from_env(Limits(max_calls=args.max_calls, max_tokens=2500, max_seconds=args.max_seconds))
    result = execute_task(task, Fixed(args.protocol), client, max_attempts=1, max_switches=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"task_id": result["task_id"], "success": result["success"], "tokens": result["tokens"], "calls": result["calls"], "wall_s": result["wall_s"], "error": result["error"], "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
