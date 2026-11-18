"""Collect the MBPP Python signal matrix using pilot_common.

This collector intentionally keeps the experiment-specific orchestration here and
delegates task execution to pilot_common.run_task/PythonTaskAdapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).absolute().parent))

from data import load_splits, sha256_file  # noqa: E402
from sandbox import run_code_with_asserts  # noqa: E402
from pilot_common import (  # noqa: E402
    AppendOnlyJournal, HarnessConfig, ModelConfig, OpenAICompatibleClient,
    PythonTaskAdapter, completed_pairs, run_task,
)

PATTERNS = ("direct", "self_refine", "verifier_loop", "memory_assisted")
MAIN_SEED = 20261008
REPEAT_SEEDS = (20261009, 20261010)
MODEL = os.environ.get("PILOT_MODEL", "ministral-3-8b-instruct-2512")
BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234")
MODEL_HASH = os.environ.get("PILOT_MODEL_ARTIFACT_HASH", "")


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hash() -> str:
    paths = [Path(__file__), ROOT / "pilot_common" / "runner.py", ROOT / "pilot_common" / "adapters.py"]
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def select_tasks():
    splits = load_splits()
    all_nonheldout = [("search", t) for t in splits["train"]] + [("validation", t) for t in splits["val"]]
    selected = all_nonheldout[:40]
    if len(selected) != 40 or len({t["task_id"] for _, t in selected}) != 40:
        raise RuntimeError("could not construct 40 disjoint non-heldout tasks")
    selected_ids = {t["task_id"] for _, t in selected}
    memory = [t for _, t in all_nonheldout if t["task_id"] not in selected_ids]
    if not memory:
        raise RuntimeError("memory search pool is empty")
    return selected, memory


def task_features(task: dict) -> dict:
    prompt = task["prompt"]
    return {
        "n_words": len(prompt.split()),
        "n_chars": len(prompt),
        "n_visible_asserts": len(task.get("test_list", [])),
        "has_edge_case": int(any(x in prompt.lower() for x in ("empty", "edge", "none", "zero"))),
    }


class MemoryBank:
    def __init__(self, items, k=2):
        self.items, self.k = items, k

    def retrieve(self, prompt):
        q = set(prompt.lower().split())
        scored = []
        for item in self.items:
            words = set(item["prompt"].lower().split())
            scored.append((len(q & words) / (len(q | words) or 1), -int(item["task_id"]), item))
        scored.sort(reverse=True)
        return [{"task_id": x["task_id"], "prompt": x["prompt"], "code": x["code"],
                 "test_list": [x["test_list"][0]]} for _, _, x in scored[:self.k]]


def verifier(task, answer):
    ok, passed, error, _ = run_code_with_asserts(answer, task["test_imports"], task["test_list"], 10.0)
    return {"success": ok, "n_asserts_passed": passed, "n_asserts": len(task["test_list"]), "error": error}


def make_config(pattern, seed, hsource):
    model = ModelConfig(backend="openai_compatible", model=MODEL, base_url=BASE_URL,
                        temperature=0.0, seed=seed, max_tokens=512, num_ctx=4096, timeout_s=1800.0)
    prompts = {
        "direct": "Write one correct, self-contained Python function for the task. Return only code.",
        "self_refine": "Write one correct, self-contained Python function. Return only code.",
        "verifier_loop": "Write one correct, self-contained Python function. Return only code.",
        "memory_assisted": "Write one correct, self-contained Python function. Use memory only as hints. Return only code.",
    }
    return HarnessConfig(system_prompt=prompts[pattern], pattern=pattern,
                         max_repairs=1 if pattern == "verifier_loop" else 0,
                         memory_k=2 if pattern == "memory_assisted" else 0,
                         use_visible_verifier=True, seed=seed, model=model,
                         harness_source_hash=hsource,
                         tool_policy={"verifier": "sandbox_python_all_asserts", "memory": "search_pool_only"})


def public_record(task, split, seed, pattern, tr, cfg, model_id, infra_error=None):
    traj = tr.to_dict()
    return {
        "task_id": str(task["task_id"]), "task_template": "mbpp-sanitized",
        "split": split, "seed": seed, "harness": pattern, "model": model_id,
        "success": int(bool(tr.success)) if not infra_error else 0,
        "tokens": tr.total_prompt_tokens + tr.total_completion_tokens,
        "tool_calls": tr.tool_calls + (1 if pattern == "memory_assisted" else 0),
        "latency_s": tr.latency_s, "task_features": task_features(task),
        "trajectory": traj["steps"], "config_hash": cfg.config_hash,
        "source_hash": cfg.harness_source_hash,
        "infrastructure_error": infra_error,
    }


def load_done(outcomes: Path):
    done = set()
    if outcomes.exists():
        for line in outcomes.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("infrastructure_error"):
                    done.add((str(r["task_id"]), r["harness"], int(r["seed"])))
    return done


def check_smoke(rows, expected_tasks):
    errors = []
    for r in rows:
        if r.get("infrastructure_error"):
            errors.append(f"{r['harness']}: infrastructure error {r['infrastructure_error']}")
        steps = r["trajectory"]
        models = [s for s in steps if s["kind"] == "model"]
        verifiers = [s for s in steps if s["kind"] == "verifier"]
        hidden = [str(t) for s in models for m in s.get("messages", []) for t in expected_tasks[r["task_id"]].get("test_list", [])[1:] if t in m.get("content", "")]
        expected_models = {"direct": 1, "self_refine": 2, "verifier_loop": (1, 2), "memory_assisted": 1}[r["harness"]]
        if len(models) not in (expected_models if isinstance(expected_models, tuple) else (expected_models,)):
            errors.append(f"{r['harness']}: model calls={len(models)}")
        if len(verifiers) < 1:
            errors.append(f"{r['harness']}: no final verifier")
        if hidden:
            errors.append(f"{r['harness']}: hidden assert leakage")
        if not r.get("config_hash") or not r.get("source_hash"):
            errors.append(f"{r['harness']}: missing hashes")
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="20261008")
    ap.add_argument("--out-root", default=str(ROOT / "runs"))
    ap.add_argument("--skip-main", action="store_true")
    a = ap.parse_args(argv)
    selected, memory_items = select_tasks()
    run_dir = Path(a.out_root) / f"python_signal_{a.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path, trajectories_path = run_dir / "outcomes.jsonl", run_dir / "trajectories.jsonl"
    journal = AppendOnlyJournal(trajectories_path)
    hsource = source_hash()
    manifest = {"dataset": "google-research-datasets/mbpp:sanitized", "data_sha256": sha256_file(Path(__file__).parent / "data" / "mbpp_sanitized.json"),
                "selection_rule": "pilot frozen train then val order, first 40", "tasks": [{"task_id": str(t["task_id"]), "split": s} for s, t in selected],
                "task_ids": [str(t["task_id"]) for _, t in selected], "task_ids_sha256": canonical_hash([str(t["task_id"]) for _, t in selected]),
                "memory_task_ids": [str(t["task_id"]) for t in memory_items], "heldout_included": False}
    (run_dir / "task_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    model_id = f"lmstudio/{MODEL}"
    base_config = {"backend": "lmstudio", "endpoint": os.environ.get("LM_STUDIO_CHAT_URL", BASE_URL.rstrip("/") + "/v1/chat/completions"),
                   "model": MODEL, "model_artifact_hash": MODEL_HASH or None, "temperature": 0.0, "max_tokens": 512, "num_ctx": 4096,
                   "main_seed": MAIN_SEED, "repeat_seeds": list(REPEAT_SEEDS), "patterns": list(PATTERNS), "source_hash": hsource,
                   "task_manifest_hash": canonical_hash(manifest), "command": "python pilot/python_signal_collect.py --run-id " + a.run_id,
                   "platform": platform.platform(), "python": sys.version}
    (run_dir / "config.json").write_text(json.dumps(base_config, ensure_ascii=False, indent=2), encoding="utf-8")
    done = load_done(outcomes_path)
    memory = MemoryBank(memory_items)
    client_cache = {}

    def run_one(split, task, pattern, seed):
        cfg = make_config(pattern, seed, hsource)
        key = (str(task["task_id"]), pattern, seed)
        if key in done:
            return "skip", key
        try:
            client = client_cache.setdefault(seed, OpenAICompatibleClient(cfg.model))
            tr = run_task(task, cfg, client, lambda answer, _task=task: verifier(_task, answer),
                          memory=memory, adapter=PythonTaskAdapter())
            infra = tr.error_type if tr.error_type else None
            record = public_record(task, split, seed, pattern, tr, cfg, model_id, infra)
        except Exception as exc:
            record = {"task_id": str(task["task_id"]), "task_template": "mbpp-sanitized", "split": split, "seed": seed,
                      "harness": pattern, "model": model_id, "success": 0, "tokens": 0, "tool_calls": 0, "latency_s": 0,
                      "task_features": task_features(task), "trajectory": [], "config_hash": cfg.config_hash,
                      "source_hash": hsource, "infrastructure_error": f"{type(exc).__name__}: {exc}"}
        with outcomes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        journal.append(record["trajectory"] and {"record_type": "trajectory", **record} or record)
        done.add(key)
        return "done", record

    smoke_tasks = selected[:2]
    smoke_rows = []
    for split, task in [(s, t) for s, t in selected[:2]]:
        for pattern in PATTERNS:
            status, row = run_one(split, task, pattern, MAIN_SEED)
            if status == "done": smoke_rows.append(row)
    smoke_errors = check_smoke(smoke_rows, {str(t["task_id"]): t for _, t in smoke_tasks})
    smoke_report = ["# Smoke report", "", f"Run: `python_signal_{a.run_id}`", f"Rows executed: {len(smoke_rows)}", f"Errors: {len(smoke_errors)}", ""]
    smoke_report += [f"- {x}" for x in smoke_errors] or ["- PASS: call counts, final verifier, no hidden asserts, and hashes"]
    (run_dir / "smoke_report.md").write_text("\n".join(smoke_report) + "\n", encoding="utf-8")
    if smoke_errors:
        raise SystemExit("smoke failed; main collection was not started: " + "; ".join(smoke_errors))
    if a.skip_main:
        return
    repeat_ids = {str(t["task_id"]) for _, t in selected[:20]}
    jobs = [(s, t, p, seed) for s, t in selected for p in PATTERNS for seed in ((MAIN_SEED, *REPEAT_SEEDS) if str(t["task_id"]) in repeat_ids else (MAIN_SEED,))]
    for i, (split, task, pattern, seed) in enumerate(jobs, 1):
        status, _ = run_one(split, task, pattern, seed)
        if i % 10 == 0 or i == len(jobs): print(f"[{i}/{len(jobs)}] {status}", flush=True)
    report = {"completed_pairs": len(load_done(outcomes_path)), "smoke_errors": smoke_errors, "heldout_evaluated": False,
              "repeat_tasks": sorted(repeat_ids), "note": "temperature=0 repeats are not independent stochastic measurements if identical"}
    (run_dir / "integrity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "README.md").write_text("# Python signal collector\n\nRun with:\n\n```powershell\n$env:PILOT_BACKEND='lmstudio'\n$env:PILOT_MODEL='" + MODEL + "'\npython pilot/python_signal_collect.py --run-id " + a.run_id + "\n```\n\nResume is automatic: completed `(task_id, pattern, seed)` rows are skipped. Held-out evaluation and policy training were not run.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
