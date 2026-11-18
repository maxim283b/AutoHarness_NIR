"""GEPA optimization of the two-prompt harness (pilot_selection.md 3.5). GEPA itself is used unchanged.

    python run_optimize.py --run-id gepa_s0 --seed 0 --max-metric-calls 400 \
        --reflection-lm ollama_chat/qwen2.5-coder:7b

Re-running with the same --run-id resumes from runs/<run_id>/gepa_state (GEPA run_dir).
Writes runs/<run_id>/:
    config.json             all settings, model digests, versions
    trajectories.jsonl      every target-LM step during optimization (phase=optimize, split=train|val)
    reflection_calls.jsonl  every reflection (teacher) call: prompt, response, tokens, latency
    gepa_state/             GEPA's own run_dir (state, candidates, resume)
    candidates.json         all candidates with val scores and parents
    best_candidate.json     result.best_candidate (selected on valset)
    summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import gepa
import litellm

from data import DEFAULT_MANIFEST, load_split
from harness_adapter import HarnessConfig, MBPPHarnessAdapter, candidate_hash
from run_eval import GEPA_COMMIT, _ver, ollama_info

HERE = Path(__file__).resolve().parent


class LoggedReflectionLM:
    """Reflection LM callable ((str | messages) -> str) that logs every call. Conforms to GEPA's LanguageModel."""

    def __init__(self, model: str, log_path: Path, api_base: str | None, **kwargs):
        self.model, self.log_path, self.api_base, self.kwargs = model, log_path, api_base, kwargs
        self.total_cost = 0.0  # local model: no monetary cost; present so GEPA doesn't wrap us
        self.prompt_tokens = self.completion_tokens = self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, prompt):
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        t0 = time.perf_counter()
        resp = litellm.completion(model=self.model, messages=messages, api_base=self.api_base, **self.kwargs)
        dt = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        u = resp.usage
        with self._lock:
            self.calls += 1
            self.prompt_tokens += u.prompt_tokens or 0
            self.completion_tokens += u.completion_tokens or 0
            with self.log_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(), "model": self.model,
                    "messages": messages, "response": text, "finish_reason": resp.choices[0].finish_reason,
                    "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "latency_s": dt,
                }, ensure_ascii=False) + "\n")
        return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-metric-calls", type=int, default=400)
    ap.add_argument("--reflection-lm", default="ollama_chat/qwen2.5-coder:7b")
    ap.add_argument("--reflection-max-tokens", type=int, default=2048)
    ap.add_argument("--reflection-num-ctx", type=int, default=8192)
    ap.add_argument("--minibatch", type=int, default=3)
    ap.add_argument("--api-base", default=HarnessConfig.api_base)
    ap.add_argument("--candidate", type=Path, default=HERE / "seed_harness.json")
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    train, val = load_split("train"), load_split("val")

    cfg = HarnessConfig(api_base=args.api_base, seed=0)  # target decoding identical to baseline
    adapter = MBPPHarnessAdapter(cfg, log_path=run_dir / "trajectories.jsonl", run_id=args.run_id, phase="optimize", split="train|val")
    refl_kwargs = {"max_tokens": args.reflection_max_tokens, "num_ctx": args.reflection_num_ctx, "seed": args.seed,
                   "timeout": 1800, "num_retries": 3}
    reflection_lm = LoggedReflectionLM(args.reflection_lm, run_dir / "reflection_calls.jsonl", args.api_base, **refl_kwargs)

    gepa_kwargs = dict(
        reflection_minibatch_size=args.minibatch,
        candidate_selection_strategy="pareto",
        module_selector="round_robin",
        max_metric_calls=args.max_metric_calls,
        cache_evaluation=True,
        seed=args.seed,
        display_progress_bar=False,
    )
    teacher = ollama_info(HarnessConfig(task_lm=args.reflection_lm, api_base=args.api_base))
    config = {
        "run_id": args.run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "seed_candidate": seed_candidate,
        "seed_candidate_hash": candidate_hash(seed_candidate),
        "harness": cfg.to_dict(),
        "target_model": ollama_info(cfg),
        "reflection_lm": args.reflection_lm,
        "reflection_kwargs": refl_kwargs,
        "reflection_model": teacher,
        "teacher_label": "local teacher" if args.reflection_lm.startswith("ollama") else "api teacher",
        "gepa_optimize_kwargs": gepa_kwargs,
        "trainset_task_ids": [t["task_id"] for t in train],
        "valset_task_ids": [t["task_id"] for t in val],
        "splits_manifest_sha256": json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))["manifest_sha256"],
        "framework": {"gepa": _ver("gepa"), "gepa_commit": GEPA_COMMIT, "litellm": _ver("litellm")},
        "python": sys.version,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    t0 = time.perf_counter()
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=reflection_lm,
        run_dir=str(run_dir / "gepa_state"),
        **gepa_kwargs,
    )
    wall = time.perf_counter() - t0

    best = result.best_candidate
    (run_dir / "best_candidate.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    cands = [
        {"idx": i, "hash": candidate_hash(c), "val_score": result.val_aggregate_scores[i],
         "parents": result.parents[i], "candidate": c}
        for i, c in enumerate(result.candidates)
    ]
    (run_dir / "candidates.json").write_text(json.dumps(cands, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "run_id": args.run_id,
        "n_candidates": len(result.candidates),
        "best_idx": result.best_idx,
        "best_val_score": result.val_aggregate_scores[result.best_idx],
        "seed_val_score": result.val_aggregate_scores[0],
        "best_candidate_hash": candidate_hash(best),
        "total_metric_calls": result.total_metric_calls,
        "reflection_calls": reflection_lm.calls,
        "reflection_prompt_tokens": reflection_lm.prompt_tokens,
        "reflection_completion_tokens": reflection_lm.completion_tokens,
        "wall_time_s": wall,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
