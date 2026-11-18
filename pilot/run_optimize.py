"""GEPA optimization of the harness prompts for one arm.

  python run_optimize.py --run-id gepa_std_s0 --arm standard
  python run_optimize.py --run-id gepa_con_s0 --arm contrastive --budget-from runs/gepa_std_s0/cost.json

--budget-from caps target-model tokens and teacher tokens at the values the reference arm actually used, in
addition to the identical max_metric_calls (task rollouts).
"""

import argparse
import difflib
import json
import platform
import subprocess
import time
from pathlib import Path

import gepa

from data import load_splits
from harness import TARGET_MODEL, TARGET_OPTIONS
from harness_adapter import MBPPHarnessAdapter
from llm import TeacherLM, TokenLedger, make_chat_client

HERE = Path(__file__).parent
TEACHER_MODEL = "qwen2.5-coder:7b"
TEACHER_OPTIONS = {"temperature": 0.7, "seed": 0, "num_ctx": 8192, "num_predict": 1024}


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


class PatchLogger:
    """Records every proposed patch with its accept/reject decision (GEPACallback)."""

    def __init__(self, run_dir: Path, ledger: TokenLedger):
        self.f_patches = run_dir / "patches.jsonl"
        self.f_events = run_dir / "events.jsonl"
        self.ledger = ledger
        self.pending: dict[int, dict] = {}
        self.t0 = time.time()

    def _event(self, kind, **kw):
        rec = {"event": kind, "t": round(time.time() - self.t0, 1), "ledger": self.ledger.snapshot(), **kw}
        with open(self.f_events, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def _patch(self, it, decision, **kw):
        p = self.pending.pop(it, None)
        if p is None:
            return
        rec = {"iteration": it, "decision": decision, **p, **kw}
        with open(self.f_patches, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def on_proposal_start(self, e):
        self.pending[e["iteration"]] = {"parent_candidate": e["parent_candidate"], "components": e["components"],
                                        "reflective_dataset": e["reflective_dataset"]}

    def on_proposal_end(self, e):
        p = self.pending.setdefault(e["iteration"], {})
        p["new_instructions"] = e["new_instructions"]
        p["raw_lm_outputs"] = e["raw_lm_outputs"]
        parent = p.get("parent_candidate", {})
        p["diff"] = {c: "".join(difflib.unified_diff(parent.get(c, "").splitlines(True), t.splitlines(True),
                                                     "parent/" + c, "child/" + c))
                     for c, t in e["new_instructions"].items()}

    def on_candidate_accepted(self, e):
        self._patch(e["iteration"], "accepted", new_candidate_idx=e["new_candidate_idx"],
                    minibatch_new_score=e["new_score"], parent_ids=list(e["parent_ids"]))
        self._event("accepted", iteration=e["iteration"], idx=e["new_candidate_idx"], score=e["new_score"])

    def on_candidate_rejected(self, e):
        self._patch(e["iteration"], "rejected", minibatch_old_score=e["old_score"],
                    minibatch_new_score=e["new_score"], reason=e["reason"])
        self._event("rejected", iteration=e["iteration"], old=e["old_score"], new=e["new_score"], reason=e["reason"])

    def on_valset_evaluated(self, e):
        self._event("valset", **{k: v for k, v in e.items() if k in ("iteration", "candidate_idx", "average_score",
                                                                       "is_best_program", "num_examples_evaluated",
                                                                       "scores_by_val_id")})

    def on_evaluation_skipped(self, e):
        self._event("skipped", iteration=e["iteration"], reason=e["reason"])

    def on_iteration_end(self, e):
        self.ledger.save(self.f_events.parent / "ledger.json")  # survives restarts (resume from gepa_state)
        self._patch(e["iteration"], "no_decision")  # e.g. proposal failed to parse
        self._event("iteration_end", iteration=e["iteration"], proposal_accepted=e.get("proposal_accepted"))

    def on_budget_updated(self, e):
        pass

    def on_error(self, e):
        self._event("error", **{k: str(v) for k, v in e.items()})


class TokenBudgetStopper:
    def __init__(self, ledger: TokenLedger, max_target: int | None, max_teacher: int | None):
        self.ledger, self.max_target, self.max_teacher = ledger, max_target, max_teacher
        self.reason = None

    def __call__(self, state) -> bool:
        if self.max_target is not None and self.ledger.total("target") >= self.max_target:
            self.reason = "target_token_budget"
        elif self.max_teacher is not None and self.ledger.total("teacher") >= self.max_teacher:
            self.reason = "teacher_token_budget"
        return self.reason is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arm", choices=["standard", "contrastive"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-metric-calls", type=int, default=400)
    ap.add_argument("--budget-from", default=None, help="cost.json of the reference arm (caps tokens)")
    a = ap.parse_args()

    run_dir = HERE / "runs" / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    train, val = splits["train"], splits["val"]
    split_of = {t["task_id"]: "train" for t in train} | {t["task_id"]: "val" for t in val}

    ledger = TokenLedger()
    ledger.load(run_dir / "ledger.json")  # resume: continue counting from the last completed iteration
    target = make_chat_client(TARGET_MODEL, TARGET_OPTIONS, ledger, "target")
    teacher = TeacherLM(make_chat_client(TEACHER_MODEL, TEACHER_OPTIONS, ledger, "teacher"), run_dir / "teacher_calls.jsonl")
    ctx = {"run_id": a.run_id, "arm": a.arm, "phase": "optimize", "split_of": split_of}
    adapter = MBPPHarnessAdapter(target, run_dir / "trajectories.jsonl", ctx, {t["task_id"] for t in train}, a.arm)

    max_target = max_teacher = None
    if a.budget_from:
        ref = json.loads(Path(a.budget_from).read_text(encoding="utf-8"))
        max_target, max_teacher = ref["optimize"]["target"]["total_tokens"], ref["optimize"]["teacher"]["total_tokens"]
    stopper = TokenBudgetStopper(ledger, max_target, max_teacher)
    seed_candidate = json.loads((HERE / "seed_harness.json").read_text(encoding="utf-8"))

    config = {
        "run_id": a.run_id, "arm": a.arm, "seed": a.seed, "max_metric_calls": a.max_metric_calls,
        "max_target_tokens": max_target, "max_teacher_tokens": max_teacher,
        "target": {"model": TARGET_MODEL, "options": TARGET_OPTIONS},
        "teacher": {"model": TEACHER_MODEL, "options": TEACHER_OPTIONS},
        "reflection_minibatch_size": 3, "candidate_selection_strategy": "pareto", "module_selector": "round_robin",
        "cache_evaluation": True, "gepa_version": getattr(gepa, "__version__", "?"), "gepa_commit": "fb1ed58",
        "seed_candidate": seed_candidate, "python": platform.python_version(),
        "splits_manifest": json.loads((HERE / "splits_manifest.json").read_text(encoding="utf-8")),
    }
    try:
        config["ollama_models"] = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    except Exception:
        pass
    jdump(run_dir / "config.json", config)

    t0 = time.time()
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=teacher,
        max_metric_calls=a.max_metric_calls,
        stop_callbacks=[stopper],
        reflection_minibatch_size=3,
        candidate_selection_strategy="pareto",
        module_selector="round_robin",
        run_dir=str(run_dir / "gepa_state"),
        seed=a.seed,
        cache_evaluation=True,
        callbacks=[PatchLogger(run_dir, ledger)],
        display_progress_bar=False,
    )
    wall = time.time() - t0

    jdump(run_dir / "best_candidate.json", result.best_candidate)
    jdump(run_dir / "all_candidates.json", [
        {"idx": i, "candidate": c, "val_score": s, "parents": result.parents[i]}
        for i, (c, s) in enumerate(zip(result.candidates, result.val_aggregate_scores))])
    jdump(run_dir / "contrast_pairs.json", adapter.contrast_log)
    cost = {
        "optimize": ledger.snapshot(),
        "metric_calls": result.total_metric_calls,
        "wall_seconds": round(wall, 1),
        "stop_reason": stopper.reason or "max_metric_calls_or_other",
        "n_candidates": len(result.candidates),
        "best_idx": result.best_idx,
        "best_val_score": result.val_aggregate_scores[result.best_idx],
        "seed_val_score": result.val_aggregate_scores[0],
        "usd": 0.0,
        "usd_note": "both models run locally via Ollama on CPU; cost is reported in tokens and CPU wall time",
    }
    jdump(run_dir / "cost.json", cost)
    print(json.dumps(cost, indent=1))


if __name__ == "__main__":
    main()
