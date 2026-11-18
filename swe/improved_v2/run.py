"""Frozen, restartable H0/H1/H2 SWE-bench Pro V1 campaign."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parent
SPEC = Path(__file__).resolve().parent
OUT = ROOT / "results/autosaddler_reproduction/improved_v2_campaign"
FREEZE = SPEC / "freeze.json"
IDS = {"H1": "swe-improved-v2-h1-20261112", "H2": "swe-improved-v2-h2-20261112"}
FINAL_ID = "swe-improved-v2-final-20261112"
ROLES = ("optimizer", "target_search", "target_final_baseline", "target_final_optimized")
LIMITS = {role: 1000.0 for role in ROLES}
REQUESTS = {role: 20000 for role in ROLES}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def sources():
    base = MAIN / "third_party/AutoSaddler/src"
    paths = list((ROOT / "experiments/autosaddler_reproduction").glob("*.py"))
    paths += list(base.rglob("*.py"))
    paths += [SPEC / "run.py", SPEC / "supervise.py", SPEC / "prepare_plan.py", ROOT / "fullsource/image-digests.json",
              ROOT / "fullsource/selected-manifest.json", ROOT / "fullsource/prepared-images.json",
              MAIN / "third_party/experiment_data/swe_bench_pro_v1.jsonl"]
    return {str(path.relative_to(MAIN)): sha(path) for path in sorted(paths)}


def freeze():
    from experiments.autosaddler_reproduction.engine import prepare_seed
    from experiments.autosaddler_reproduction.source_build import candidate_files
    if FREEZE.exists():
        return verify()
    if any((OUT / label / "gateway/requests.jsonl").exists() for label in (*IDS, "final")):
        raise RuntimeError("Inference already began before preregistration")
    manifest = json.loads((SPEC / "manifest.json").read_text())
    selected = {name: manifest[name] for name in ("train", "development", "test")}
    if {key: len(value) for key, value in selected.items()} != {"train": 24, "development": 16, "test": 40}:
        raise RuntimeError("Split size drift")
    if len(set().union(*(set(value) for value in selected.values()))) != 80:
        raise RuntimeError("Split overlap")
    if set(selected["test"]) & set(manifest["previously_exposed_test_ids"]):
        raise RuntimeError("Prior test exposure leaked")
    if sha(MAIN / "third_party/experiment_data/swe_bench_pro_v1.jsonl") != manifest["source_jsonl_sha256"]:
        raise RuntimeError("Corpus drift")
    original = json.loads((ROOT / "fullsource/selected-manifest.json").read_text())
    if set().union(*(set(value) for value in selected.values())) != set().union(*(set(original[key]) for key in selected)):
        raise RuntimeError("Selected IDs outside certified image universe")
    image_plan = json.loads((ROOT / "fullsource/image-digests.json").read_text())
    if set(image_plan["resolved"]) != set().union(*(set(value) for value in selected.values())):
        raise RuntimeError("Image plan drift")
    for label in IDS:
        import yaml
        config = yaml.safe_load((SPEC / f"{label}.yaml").read_text())
        settings = config["scenario"]["settings"]
        if (config["optimization"]["budget"]["max_iterations"] != 6
            or config["optimization"]["task_selection"]["batch_size"] != 8
            or config["provider"]["settings"]["model"] != "opencode-go/glm-5.3"
            or settings["target_model"] != "opencode-go/qwen3.7-plus"
            or settings["improved_autosaddler_v2"] != (label == "H2")
            or settings["fullsource_manifest"] != "improved_v2/manifest.json"):
            raise RuntimeError(f"{label} configuration drift")
    seed, revision = prepare_seed()
    frozen = {"schema": "improved-autosaddler-v2-freeze/v1", "frozen_at_utc": now(),
              "run_ids": IDS, "final_run_id": FINAL_ID, "selected": selected,
              "source_sha256": sources(), "plan_sha256": {p.name: sha(p) for p in
                  (SPEC / "manifest.json", SPEC / "H1.yaml", SPEC / "H2.yaml")},
              "seed_revision": revision, "seed_files": candidate_files(seed),
              "optimizer": "opencode-go/glm-5.3", "target": "opencode-go/qwen3.7-plus",
              "previous_exposure": manifest["prior_observations_disclosure"],
              "final_schedule": "task-adjacent SHA256 permutation of H0/H1/H2; each task once per label",
              "external_model_calls_before_freeze": 0}
    save(FREEZE, frozen)
    return frozen


def verify():
    frozen = json.loads(FREEZE.read_text())
    if frozen["source_sha256"] != sources():
        raise RuntimeError("Frozen source changed")
    if frozen["plan_sha256"] != {p.name: sha(p) for p in
                                  (SPEC / "manifest.json", SPEC / "H1.yaml", SPEC / "H2.yaml")}:
        raise RuntimeError("Frozen plan changed")
    from experiments.autosaddler_reproduction.engine import prepare_seed
    from experiments.autosaddler_reproduction.source_build import candidate_files
    seed, revision = prepare_seed()
    if revision != frozen["seed_revision"] or candidate_files(seed) != frozen["seed_files"]:
        raise RuntimeError("H0 changed")
    return frozen


def gateway(path):
    from experiments.autosaddler_reproduction.scoped_gateway import ScopedGoGateway
    return ScopedGoGateway(path, allowance_limit=4000.0, request_limit=80000,
                           role_limits=LIMITS, role_request_limits=REQUESTS,
                           require_context_policy=True, optimizer_model="glm-5.3", quota_ceiling=100)


def optimize(label):
    from experiments.autosaddler_reproduction.engine import build
    from experiments.autosaddler_reproduction.complete_engine import require_complete_search
    runtime = build(SPEC / f"{label}.yaml", IDS[label])
    with gateway(OUT / label / "gateway") as go:
        try:
            result = runtime.engine.run()
            proof = require_complete_search(runtime.store, 6)
            save(OUT / label / "complete.json", {**proof, "candidate_id": result.selected_candidate_id,
                "development_score": result.development_score, "run_id": IDS[label]})
        finally:
            save(OUT / label / "accounting.json", go.accounting_summary())


def candidate(label):
    from autosaddler.v2.core.run_state import RunState
    from experiments.autosaddler_reproduction.engine import build
    from experiments.autosaddler_reproduction.complete_engine import require_complete_search
    runtime = build(SPEC / f"{label}.yaml", IDS[label])
    require_complete_search(runtime.store, 6)
    state = RunState.replay(runtime.store.events())
    if not state.optimization_completed or not state.selected_candidate_id:
        raise RuntimeError(f"{label} selection incomplete")
    completion = json.loads((OUT / label / "complete.json").read_text())
    if completion["candidate_id"] != state.selected_candidate_id:
        raise RuntimeError(f"{label} candidate provenance mismatch")
    return runtime, state.candidates[state.selected_candidate_id]


def result_valid(result, expected, role):
    if any(result.get(key) != value for key, value in expected.items()):
        return False
    usage = result.get("usage", {}).get(role, {})
    return (result.get("valid_evaluation") is True and result.get("score") in (0, 0.0, 1, 1.0)
            and not result.get("synthetic", False) and result.get("agent_cleanup_verified") is True
            and result.get("generation_container_removed_before_grading") is True
            and result.get("grading_image_id") == expected.get("grading_image_id")
            and usage.get("usage_complete") is True
            and usage.get("calls", 0) >= 1 and usage.get("total_tokens") is not None)


def final():
    from autosaddler.v2.core.domain import Case
    from experiments.autosaddler_reproduction import runner
    from experiments.autosaddler_reproduction.opencode_runtime import candidate_hash
    h1_runtime, h1 = candidate("H1")
    h2_runtime, h2 = candidate("H2")
    baseline = h1_runtime.scenario.harness_space.seed()
    if baseline.candidate_id != h2_runtime.scenario.harness_space.seed().candidate_id:
        raise RuntimeError("H0 seed differs between arms")
    candidates = {"H0": baseline, "H1": h1, "H2": h2}
    manifest = json.loads((SPEC / "manifest.json").read_text())
    order = sorted(manifest["test"], key=lambda task: hashlib.sha256(
        f"20261112:swe_bench_pro:test:{task}".encode()).hexdigest())
    if order[:16] != manifest["test_fallback_first_16"]:
        raise RuntimeError("Frozen fallback subset drift")
    schedule = []
    for task in order:
        labels = sorted(candidates, key=lambda label: hashlib.sha256(
            f"20261112:final:{task}:{label}".encode()).hexdigest())
        schedule.extend((task, label, labels[0]) for label in labels)
    plan = {"run_id": FINAL_ID, "test": order, "schedule": schedule,
            "candidate_ids": {label: value.candidate_id for label, value in candidates.items()},
            "manifest_sha256": sha(SPEC / "manifest.json")}
    plan_path = OUT / "final/plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Final plan drift")
    save(plan_path, plan)
    with gateway(OUT / "final/gateway") as go:
        try:
            for index, (task, label, first) in enumerate(schedule):
                role = "target_final_baseline" if label == "H0" else "target_final_optimized"
                path = OUT / "final/results" / label / hashlib.sha256(task.encode()).hexdigest()[:16]
                result_path = path / "result.json"
                expected = {"run_id": FINAL_ID, "label": label, "benchmark": "swe_bench_pro",
                            "task_id": task, "candidate_id": candidates[label].candidate_id,
                            "model": "opencode-go/qwen3.7-plus", "execution_index": index,
                            "planned_first_label": first, "role": role}
                preflight = ROOT / "results/autosaddler_reproduction/fullsource_image_preflight" / (hashlib.sha256(task.encode()).hexdigest()[:20] + ".json")
                certificate = json.loads(preflight.read_text())
                if certificate.get("task_id") != task or certificate.get("passed") is not True:
                    raise RuntimeError("Task image certificate drift")
                expected["grading_image_id"] = certificate["image_id"]
                runtime = h1_runtime if label != "H2" else h2_runtime
                materialized = runtime.scenario.harness_space.materialize(candidates[label], "evaluate")
                try:
                    expected["candidate_hash"] = candidate_hash(materialized.root)
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                        if not result_valid(result, expected, role):
                            raise RuntimeError(f"Incomplete or mismatched final result: {result_path}")
                        continue
                    if (path / "attempt.json").exists():
                        raise RuntimeError(f"Interrupted final attempt requires artifact reconciliation: {path}")
                    save(path / "attempt.json", {**expected, "started_at_utc": now()})
                    case = Case("swe_bench_pro:" + task, "test", {"benchmark": "swe_bench_pro", "task_id": task})
                    result = runner.run_case(case, materialized.root, path,
                                             runtime.config.scenario.settings, FINAL_ID, role)
                finally:
                    materialized.release()
                result.update(expected)
                result["completed_at_utc"] = now()
                if not result_valid(result, expected, role):
                    save(path / "invalid-result.json", result)
                    raise RuntimeError(f"Official evaluation incomplete: {task} {label}")
                save(result_path, result)
        finally:
            save(OUT / "final/accounting.json", go.accounting_summary())


def run():
    verify()
    lock = Path.home() / ".local/share/autoharness/go-plus-owner.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for label in IDS:
                save(OUT / "status.json", {"stage": label, "status": "running", "updated_at_utc": now()})
                optimize(label)
            save(OUT / "status.json", {"stage": "final", "status": "running", "updated_at_utc": now()})
            final()
            save(OUT / "status.json", {"stage": "done", "status": "complete", "updated_at_utc": now()})
        except BaseException as error:
            save(OUT / "status.json", {"status": "stopped", "error_type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc(), "updated_at_utc": now()})
            raise
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "verify", "run"))
    action = parser.parse_args().action
    if action == "freeze":
        print(json.dumps(freeze(), ensure_ascii=False))
    elif action == "verify":
        print(json.dumps({"verified": True, "freeze_sha256": sha(FREEZE)})) if verify() else None
    else:
        run()
