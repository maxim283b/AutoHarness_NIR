from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.controllers.policies import Fixed, Rule, LLMController, LinUCB, RandomPolicy, features
from src.execution.llm import Client, Limits
from src.execution.runner import execute_task
from src.execution.state import PROTOCOLS, TaskState
from src.storage.jsonl import read_jsonl, write_jsonl


def reward(row: dict, max_tokens: int) -> float:
    # Predeclared tradeoff: at most 0.1 success point for exhausting the entire task budget.
    return float(row["success"]) - 0.1 * min(1.0, row["tokens"] / max_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--real-budget-approved", action="store_true", help="Explicit approval for paid/large real-model run")
    parser.add_argument("--limit", type=int, default=0, help="Tasks per split; 0 means full split")
    parser.add_argument("--max-calls-per-task", type=int, default=12)
    parser.add_argument("--max-tokens-per-task", type=int, default=10000)
    parser.add_argument("--max-seconds-per-task", type=float, default=120)
    parser.add_argument("--max-total-calls", type=int, default=2000)
    parser.add_argument("--max-total-tokens", type=int, default=1_500_000)
    parser.add_argument("--max-total-seconds", type=float, default=7200)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "mock_smoke")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run with identical split/model/limits")
    args = parser.parse_args()
    if args.mode == "real" and not args.real_budget_approved:
        parser.error("Real run requires explicit --real-budget-approved after reviewing estimated limits")
    if not args.resume and args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory is not empty; choose a fresh --out")
    train = read_jsonl(ROOT / "data" / "train.jsonl")[: args.limit or None]
    test = read_jsonl(ROOT / "data" / "test.jsonl")[: args.limit or None]
    n_runs = 4 * len(train) + 10 * len(test)
    print(f"Planned {n_runs} task-runs; conservative ceiling {n_runs * args.max_calls_per_task} calls, {n_runs * args.max_tokens_per_task} tokens, {n_runs * args.max_seconds_per_task:.0f} seconds. Hard global caps: {args.max_total_calls} calls, {args.max_total_tokens} tokens, {args.max_total_seconds:.0f} seconds.")
    if os.getenv("OPENAI_INPUT_USD_PER_MTOK") and os.getenv("OPENAI_OUTPUT_USD_PER_MTOK"):
        high_rate = max(float(os.environ["OPENAI_INPUT_USD_PER_MTOK"]), float(os.environ["OPENAI_OUTPUT_USD_PER_MTOK"]))
        print(f"Conservative cost ceiling if all tokens were at the higher configured rate: ${n_runs * args.max_tokens_per_task * high_rate / 1_000_000:.2f}.")
    if args.mode == "real" and (n_runs * args.max_calls_per_task > args.max_total_calls or n_runs * args.max_tokens_per_task > args.max_total_tokens):
        parser.error("Conservative ceiling exceeds global caps; reduce task count or set reviewed caps")
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    trace_path = args.out / "traces.jsonl"
    config_path = args.out / "config.json"
    all_rows: list[dict] = read_jsonl(trace_path) if args.resume and trace_path.exists() else []
    if args.resume:
        if not config_path.exists():
            parser.error("Cannot resume without config.json")
        previous = json.loads(config_path.read_text(encoding="utf8"))
        if previous["mode"] != args.mode or previous["model"] != os.getenv("OPENAI_MODEL", "scripted-mock" if args.mode == "mock" else "") or previous["train_ids"] != [t["task_id"] for t in train] or previous["test_ids"] != [t["task_id"] for t in test]:
            parser.error("Resume model, mode, or split differs from saved config")
        for key in ("max_calls_per_task", "max_tokens_per_task", "max_seconds_per_task"):
            if previous["limits"][key] != getattr(args, key):
                parser.error(f"Resume limit differs: {key}")
        for key in ("max_total_calls", "max_total_tokens", "max_total_seconds"):
            if getattr(args, key) < previous["limits"][key]:
                parser.error(f"Resume global cap cannot be reduced: {key}")
    seen = {(r["split"], r["arm"], r["task_id"]) for r in all_rows}
    total_calls = sum(r["calls"] for r in all_rows)
    total_tokens = sum(r["tokens"] for r in all_rows)
    prior_wall = sum(r["wall_s"] for r in all_rows)
    base_config = {"mode": args.mode, "model": os.getenv("OPENAI_MODEL", "scripted-mock" if args.mode == "mock" else ""), "best_fixed_train": None, "train_ids": [t["task_id"] for t in train], "test_ids": [t["task_id"] for t in test], "limits": vars(args) | {"out": str(args.out), "resume": False}, "completed": False}
    config_path.write_text(json.dumps(base_config, ensure_ascii=False, indent=2), encoding="utf8")

    def run_arm(split: str, arm: str, tasks: list[dict], policy, front_only: bool = False) -> list[dict]:
        nonlocal total_calls, total_tokens
        rows = []
        for task in tasks:
            key = (split, arm, task["task_id"])
            if key in seen:
                rows.append(next(r for r in all_rows if (r["split"], r["arm"], r["task_id"]) == key))
                continue
            if total_calls + args.max_calls_per_task > args.max_total_calls or total_tokens + args.max_tokens_per_task > args.max_total_tokens or prior_wall + time.monotonic() - start > args.max_total_seconds:
                raise RuntimeError("Global experiment budget reached; partial results saved")
            client = Client.from_env(Limits(args.max_calls_per_task, args.max_tokens_per_task, args.max_seconds_per_task), args.mode == "mock")
            row = execute_task(task, policy, client, front_only=front_only)
            row.update({"split": split, "arm": arm, "model": client.model})
            total_calls += row["calls"]
            total_tokens += row["tokens"]
            rows.append(row)
            all_rows.append(row)
            seen.add(key)
            write_jsonl(trace_path, all_rows)
            print(f"{len(all_rows)}/{n_runs} {split}/{arm} task={task['task_id']} success={row['success']} calls={row['calls']} wall={row['wall_s']:.1f}s", flush=True)
        return rows

    fixed_train = {p: run_arm("train", p, train, Fixed(p)) for p in PROTOCOLS}
    best = max(PROTOCOLS, key=lambda p: (sum(r["success"] for r in fixed_train[p]), -sum(r["tokens"] for r in fixed_train[p])))
    bandit = LinUCB()
    for p in PROTOCOLS:
        for task, row in zip(train, fixed_train[p]):
            state = TaskState(task_id=task["task_id"], prompt=task["prompt"], visible_test=task["test_list"][0], imports=task.get("test_imports", []))
            bandit.update(p, features(state, args.max_tokens_per_task), reward(row, args.max_tokens_per_task))
    for p in PROTOCOLS:
        run_arm("test", p, test, Fixed(p))
    run_arm("test", "rule", test, Rule())
    run_arm("test", "llm_controller", test, LLMController())
    run_arm("test", "bandit_trained", test, bandit)
    run_arm("test", "bandit_cold", test, LinUCB())
    run_arm("test", "random", test, RandomPolicy(173))
    run_arm("test", "rule_front_only", test, Rule(), True)
    config = base_config | {"best_fixed_train": best, "total_calls": total_calls, "total_tokens": total_tokens, "completed": True}
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"Completed; best train fixed={best}; actual {total_calls} calls, {total_tokens} tokens. Results: {args.out}")


if __name__ == "__main__":
    main()
