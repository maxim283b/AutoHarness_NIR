from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gepa

from pilot_sql.tasks import load_tasks

from .models import COMPONENTS, HarnessCandidate, digest, seed_candidate
from .optimizers import SequentialRLOptimizer, TailAwareRLOptimizer, mean_reward
from .proposer import LMStudioProposer, SharedGEPAProposer
from .sql_gepa_adapter import SQLGEPAAdapter


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("framework_original_optimizer", "random_search", "rl_optimizer", "tail_aware_rl_variant")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class ProposalStopper:
    def __init__(self, proposer: SharedGEPAProposer, limit: int):
        self.proposer, self.limit = proposer, limit

    def __call__(self, _state) -> bool:
        return self.proposer.calls >= self.limit


def task_splits() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((ROOT / "pilot_sql/splits_manifest.json").read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in load_tasks()}
    return ([tasks[x] for x in manifest["splits"]["search"][:3]],
            [tasks[x] for x in manifest["splits"]["validation"][:10]])


def rows_for(adapter: SQLGEPAAdapter, candidate: HarnessCandidate, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adapter.evaluate(tasks, candidate.components, capture_traces=False)
    return [adapter._evaluate_one(task, candidate.components) for task in tasks]


def feedback(adapter: SQLGEPAAdapter, candidate: HarnessCandidate, tasks):
    batch = adapter.evaluate(tasks, candidate.components, capture_traces=True)
    return adapter.make_reflective_dataset(candidate.components, batch, list(COMPONENTS))


def optimize(method: str, seed: int, run_dir: Path, search_tasks):
    adapter = SQLGEPAAdapter(model_seed=20261012 + seed, log_path=run_dir / "evaluations.jsonl")
    lm = LMStudioProposer(seed=20261012 + seed)
    proposal_log: list[dict[str, Any]] = []
    shared = SharedGEPAProposer(lm, proposal_log)
    base = seed_candidate()
    candidates = [base]
    if method == "framework_original_optimizer":
        result = gepa.optimize(
            seed_candidate=base.components,
            trainset=search_tasks,
            valset=search_tasks,
            adapter=adapter,
            custom_candidate_proposer=shared,
            max_metric_calls=10000,
            stop_callbacks=[ProposalStopper(shared, 15)],
            reflection_minibatch_size=3,
            candidate_selection_strategy="pareto",
            module_selector="round_robin",
            cache_evaluation=True,
            seed=seed,
            skip_perfect_score=False,
            run_dir=str(run_dir / "gepa_state" / f"seed{seed}"),
            display_progress_bar=False,
        )
        candidates.extend(HarnessCandidate(dict(c)) for c in result.candidates[1:])
        final = HarnessCandidate(dict(result.best_candidate))
        policy_state = {"official_gepa_commit": "fb1ed589fd83372caef499cffc2c73173d3b096b", "best_idx": result.best_idx}
    elif method == "random_search":
        rng = random.Random(seed)
        scored = [(mean_reward(rows_for(adapter, base, search_tasks)), base)]
        refds = feedback(adapter, base, search_tasks)
        for _ in range(15):
            component = rng.choice(COMPONENTS)
            updates = shared(base.components, refds, [component])
            child = base.apply(updates, "shared proposer random-search action")
            candidates.append(child)
            scored.append((mean_reward(rows_for(adapter, child, search_tasks)), child))
        final = max(scored, key=lambda x: (x[0], x[1].candidate_hash))[1]
        policy_state = {"algorithm": "uniform_random_over_components"}
    else:
        cls = SequentialRLOptimizer if method == "rl_optimizer" else TailAwareRLOptimizer
        optimizer = cls(seed=seed)

        class Bridge:
            calls = 0
            def propose(self, parent, component, safe_feedback, history):
                updates = shared(parent.components, {component: safe_feedback}, [component])
                self.calls = shared.calls
                child = parent.apply(updates, "shared proposer RL action")
                candidates.append(child)
                return child

        final = optimizer.run(base, Bridge(), lambda c: rows_for(adapter, c, search_tasks), steps=15)
        policy_state = optimizer.policy_state()
    if shared.calls != 15:
        raise RuntimeError(f"{method} seed {seed}: proposer calls={shared.calls}, expected 15")
    final_path = run_dir / "final_configs" / f"{method}_seed{seed}.json"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(final.public_record(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for index, candidate in enumerate(candidates):
        path = run_dir / "candidates" / f"{candidate.candidate_hash}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(candidate.public_record(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append(run_dir / "optimization_log.jsonl", {"timestamp": now(), "method": method, "seed": seed,
           "event": "optimizer_complete", "final_hash": final.candidate_hash, "proposer_calls": shared.calls,
           "target_calls": adapter.target_calls, "target_tokens": adapter.target_tokens,
           "candidate_count": len(candidates), "policy_state": policy_state, "proposal_log": proposal_log})
    return final, adapter, shared.calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_dir = ROOT / "results/final_harness_experiment" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    search, validation = task_splits()
    methods = METHODS if not args.smoke else METHODS[:1]
    seeds = (0, 1, 2) if not args.smoke else (0,)
    manifest = {"run_id": args.run_id, "started_at": now(), "framework": "GEPA",
                "framework_commit": "fb1ed589fd83372caef499cffc2c73173d3b096b",
                "methods": methods, "seeds": seeds, "search_task_ids": [x["task_id"] for x in search],
                "validation_task_ids": [x["task_id"] for x in validation], "held_out_opened": False,
                "target_model": "ministral-3-8b-instruct-2512", "proposer_model": "ministral-3-8b-instruct-2512",
                "proposer_call_limit": 15, "target_call_cap_search": 96}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frozen = []
    for method in methods:
        for seed in seeds:
            final, adapter, proposer_calls = optimize(method, seed, run_dir, search)
            frozen.append((method, seed, final, adapter, proposer_calls))
    append(run_dir / "optimization_log.jsonl", {"timestamp": now(), "event": "all_finalists_frozen"})
    for method, seed, final, adapter, proposer_calls in frozen:
        before = adapter.target_calls
        rows = rows_for(adapter, final, validation)
        append(run_dir / "optimization_log.jsonl", {"timestamp": now(), "event": "validation_complete",
               "method": method, "seed": seed, "candidate_hash": final.candidate_hash,
               "success": sum(float(x["success"]) for x in rows) / len(rows),
               "reward": mean_reward(rows), "validation_target_calls": adapter.target_calls - before,
               "proposer_calls": proposer_calls})
    print(json.dumps({"run_dir": str(run_dir), "optimizers": len(frozen), "held_out_opened": False}, indent=2))


if __name__ == "__main__":
    main()
