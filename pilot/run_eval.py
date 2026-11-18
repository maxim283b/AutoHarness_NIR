"""Evaluate one candidate harness on a split (default: held-out test).

  python run_eval.py --candidate seed_harness.json --split test --run-id baseline_test
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

from data import load_splits
from harness import TARGET_MODEL, TARGET_OPTIONS, TrajectoryLog, run_task, sha
from llm import TokenLedger, make_chat_client

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-repair", action="store_true")
    a = ap.parse_args()

    run_dir = HERE / "runs" / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cand = json.loads(Path(a.candidate).read_text(encoding="utf-8"))
    tasks = load_splits()[a.split][: a.limit]
    ledger = TokenLedger()
    client = make_chat_client(TARGET_MODEL, TARGET_OPTIONS, ledger, "target")
    log_path = run_dir / "trajectories.jsonl"
    log_path.unlink(missing_ok=True)
    log = TrajectoryLog(log_path)
    ctx = {"run_id": a.run_id, "phase": "final_eval", "split": a.split}

    rows, t0 = [], time.time()
    for i, task in enumerate(tasks):
        t = run_task(task, cand, client, log, ctx, use_repair=not a.no_repair)
        rows.append({"task_id": t["task_id"], "pass": int(t["final_pass"]), "n_asserts_passed": t["n_asserts_passed"],
                     "n_asserts": len(task["test_list"]), "repair_used": int(t["repair_used"]),
                     "repair_success": int(t["repair_used"] and t["final_pass"]),
                     "prompt_tokens": sum(s["prompt_tokens"] for s in t["steps"]),
                     "completion_tokens": sum(s["completion_tokens"] for s in t["steps"]),
                     "latency_s": round(sum(s["latency_s"] for s in t["steps"]), 2)})
        print(f"[{i + 1}/{len(tasks)}] task {t['task_id']}: pass={t['final_pass']} repair={t['repair_used']}", flush=True)

    with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    summary = {"run_id": a.run_id, "candidate_file": a.candidate, "candidate_hash": sha(cand), "candidate": cand,
               "backend": os.environ.get("PILOT_BACKEND", "lmstudio"), "target_model": TARGET_MODEL,
               "model_artifact_hash": os.environ.get("PILOT_MODEL_ARTIFACT_HASH"),
               "target_options": TARGET_OPTIONS,
               "split": a.split, "n": n, "pass_at_1": sum(r["pass"] for r in rows) / n,
               "repair_rate": sum(r["repair_used"] for r in rows) / n,
               "tokens_per_task": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows) / n,
               "ledger": ledger.snapshot(), "wall_seconds": round(time.time() - t0, 1), "no_repair": a.no_repair}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("pass_at_1", "repair_rate", "tokens_per_task", "wall_seconds")}))


if __name__ == "__main__":
    main()
