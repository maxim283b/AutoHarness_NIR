"""Evaluate a harness candidate on a fixed split; save everything needed to reproduce the run.

    python run_eval.py --candidate seed_harness.json --split test --run-id baseline
    python run_eval.py --candidate seed_harness.json --split test --no-repair --run-id direct

Writes runs/<run_id>/:
    config.json        harness config, candidate, model digest, package versions, manifest sha
    results.jsonl      one line per task: task_id, answer, verifier, tokens, latency, seed, trajectory
    results.csv        same without trajectory (flat)
    trajectories.jsonl one line per LM step (schema from pilot_selection.md 3.4)
    summary.json       pass@1 and aggregates
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import urllib.request
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from data import DEFAULT_MANIFEST, load_split
from harness_adapter import COMPONENTS, HarnessConfig, MBPPHarnessAdapter, candidate_hash

HERE = Path(__file__).resolve().parent
GEPA_COMMIT = "fb1ed58"


def _get_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    except Exception:
        return None


def ollama_info(cfg: HarnessConfig) -> dict:
    name = cfg.task_lm.split("/", 1)[1]
    tags = _get_json(cfg.api_base + "/api/tags") or {}
    model = next((m for m in tags.get("models", []) if m.get("name") == name or m.get("model") == name), None)
    return {
        "ollama_version": (_get_json(cfg.api_base + "/api/version") or {}).get("version"),
        "model_name": name,
        "model_digest": model.get("digest") if model else None,
        "model_details": model.get("details") if model else None,
    }


def _ver(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, default=HERE / "seed_harness.json")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks of the split")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=HarnessConfig.task_lm)
    ap.add_argument("--api-base", default=HarnessConfig.api_base)
    ap.add_argument("--no-repair", action="store_true", help="B0 Direct: SOLVE only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        sys.exit(f"{run_dir} exists; pass --overwrite or use another --run-id")
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in ("results.jsonl", "results.csv", "trajectories.jsonl", "summary.json"):
        (run_dir / f).unlink(missing_ok=True)

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    assert set(candidate) == set(COMPONENTS), f"candidate must have exactly {COMPONENTS}"
    tasks = load_split(args.split)
    if args.limit:
        tasks = tasks[: args.limit]

    cfg = HarnessConfig(task_lm=args.model, api_base=args.api_base, seed=args.seed, use_repair=not args.no_repair)
    adapter = MBPPHarnessAdapter(cfg, log_path=run_dir / "trajectories.jsonl", run_id=args.run_id, phase="final_eval", split=args.split)

    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    config = {
        "run_id": args.run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "split": args.split,
        "limit": args.limit,
        "task_ids": [t["task_id"] for t in tasks],
        "harness": cfg.to_dict(),
        "candidate": candidate,
        "candidate_hash": candidate_hash(candidate),
        "candidate_file": str(args.candidate),
        "splits_manifest_sha256": manifest["manifest_sha256"],
        "model": ollama_info(cfg),
        "framework": {"gepa": _ver("gepa"), "gepa_commit": GEPA_COMMIT, "litellm": _ver("litellm")},
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if config["model"]["model_digest"] is None:
        print(f"WARNING: could not read digest for {cfg.task_lm} from {cfg.api_base}", file=sys.stderr)

    fields = ["task_id", "score", "n_passed", "n_total", "verifier_error", "timed_out", "repair_used",
              "visible_ok_first", "n_lm_calls", "prompt_tokens", "completion_tokens", "latency_s", "seed",
              "candidate_hash", "harness_error"]
    t_start = time.perf_counter()
    rows = []
    with (run_dir / "results.jsonl").open("w", encoding="utf-8", newline="\n") as fj, \
         (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as fc:
        w = csv.DictWriter(fc, fieldnames=fields)
        w.writeheader()
        for i, task in enumerate(tasks, 1):
            # one task per evaluate() call so results are flushed incrementally
            r = adapter.evaluate([task], candidate, capture_traces=False).outputs[0]
            v = r["verifier"]
            fj.write(json.dumps({
                "task_id": r["task_id"], "answer": r["answer"], "verifier": v, "score": r["score"],
                "repair_used": r["repair_used"], "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"], "latency_s": r["latency_s"],
                "seed": r["seed"], "candidate_hash": r["candidate_hash"], "harness_error": r["harness_error"],
                "harness_config": cfg.to_dict(), "trajectory": r["steps"],
            }, ensure_ascii=False) + "\n")
            fj.flush()
            row = {k: r.get(k) for k in fields}
            row.update(n_passed=v["n_passed"], n_total=v["n_total"], verifier_error=v["error"], timed_out=v["timed_out"],
                       latency_s=round(r["latency_s"], 3))
            w.writerow(row)
            fc.flush()
            rows.append(r)
            print(f"[{i}/{len(tasks)}] task {r['task_id']}: {'PASS' if r['score'] else 'fail'}"
                  f"  repair={r['repair_used']}  tok={r['prompt_tokens']}+{r['completion_tokens']}  {r['latency_s']:.1f}s", flush=True)

    n = len(rows)
    summary = {
        "run_id": args.run_id,
        "split": args.split,
        "n_tasks": n,
        "pass_at_1": sum(r["score"] for r in rows) / n,
        "n_pass": int(sum(r["score"] for r in rows)),
        "visible_pass_after_solve": sum(r["visible_ok_first"] for r in rows) / n,
        "repair_triggered": sum(r["repair_used"] for r in rows),
        "repair_success": sum(r["repair_used"] and r["score"] == 1.0 for r in rows),
        "harness_errors": sum(r["harness_error"] is not None for r in rows),
        "lm_calls": sum(r["n_lm_calls"] for r in rows),
        "prompt_tokens_total": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens_total": sum(r["completion_tokens"] for r in rows),
        "tokens_per_task_mean": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows) / n,
        "lm_latency_per_task_mean_s": sum(r["latency_s"] for r in rows) / n,
        "wall_time_s": time.perf_counter() - t_start,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
