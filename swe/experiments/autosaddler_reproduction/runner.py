"""Single dispatch path for H0/H1 evaluations, independent of AutoSaddler."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .accounting import UsageLedger
from .benchmarks import selected_benchmarks
from .opencode_runtime import candidate_hash

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).parent


def agent_interactions(artifacts):
    """Only target-visible actions/results, never oracle or hidden test contents."""
    interactions = []
    for path in sorted(artifacts.rglob("delivered_events.json")):
        interactions.extend({"source": path.relative_to(artifacts).as_posix(), "environment_message": message} for message in json.loads(path.read_text(encoding="utf-8")))
    for path in sorted(artifacts.rglob("acp_events.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            update = event.get("params", {}).get("update", {})
            if update.get("sessionUpdate") in {"tool_call", "tool_call_update"}:
                interactions.append({"source": path.relative_to(artifacts).as_posix(), "event": update})
    for name in ("opencode.jsonl", "opencode.txt"):
        for path in sorted(artifacts.rglob(name)):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if value.get("type") in {"tool_use", "text", "error", "step_finish"}:
                    interactions.append({"source": path.relative_to(artifacts).as_posix(), "event": value})
    return interactions


def run_case(case, candidate, artifacts, settings, run_id, role):
    benchmark, task_id = case.payload["benchmark"], case.payload["task_id"]
    if benchmark not in selected_benchmarks(settings):
        raise ValueError("Task benchmark is outside configured experiment scope")
    from .case_selection import manifest_path
    manifest = json.loads(manifest_path(settings, benchmark).read_text())
    if task_id not in manifest[case.split]:
        raise ValueError("Task is absent from the declared split")
    if case.split == "test" and role not in {"target_final_baseline", "target_final_optimized"}:
        raise ValueError("Test tasks are forbidden during optimization")
    model = settings.get("target_model")
    if not model:
        raise RuntimeError("Target model must be configured explicitly")
    artifacts = artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    # Copy only target harness content, never Git metadata, optimizer history or evidence.
    workspace = artifacts / "target_workspace"
    if settings.get('fullsource_manifest') and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(exist_ok=True)
    for name in ("instructions.md", "opencode.json", ".opencode", "opencode-src"):
        source, dest = candidate / name, workspace / name
        if source.is_file():
            shutil.copy2(source, dest)
        elif source.is_dir():
            shutil.copytree(source, dest, dirs_exist_ok=True)
    digest = candidate_hash(workspace)
    ledger_path = ROOT / "results/autosaddler_reproduction/usage" / run_id / "requests.sqlite"
    ledger = UsageLedger(ledger_path)
    def case_rows():
        return ledger.connection.execute("SELECT request_id,role,payload FROM calls WHERE run_id=? AND benchmark=? AND task_id=?", (run_id, benchmark, task_id)).fetchall()
    previous_requests = {row[0] for row in case_rows()}
    try:
        if benchmark == "terminal_bench_2":
            from .terminal_adapter import run_task
            result = run_task(task_id, workspace, model, artifacts / "harbor", ledger, run_id, role)
        elif benchmark == "swe_bench_pro":
            from .swe_cached_images import image_lease
            with image_lease():
                from .swe_adapter import load_row
                from .swe_go_adapter import evaluate as native_evaluate
                row = load_row(task_id)
                if not settings.get("swe_native_go_optimization"):
                    raise RuntimeError("Independent SWE search requires the native Go adapter")
                if settings.get('fullsource_manifest'):
                    from .swe_cached_images import ensure_image
                    image = ensure_image(task_id, row)
                elif settings.get("swe_staged_images"):
                    from .swe_staged_images import ensure_image
                    image = ensure_image(task_id, row, settings["swe_registry_digest_file"], allow_pull=True)
                else:
                    proof_path = settings.get("swe_environment_preflight")
                    if not proof_path:
                        raise RuntimeError("SWE environment preflight is required")
                    proof = json.loads((ROOT / proof_path).read_text(encoding="utf-8"))
                    if not proof.get("passed") or proof.get("model_calls") != 0:
                        raise RuntimeError("SWE preflight did not pass without inference")
                    images = [item for item in proof.get("images", []) if item.get("task_id") == task_id]
                    if len(images) != 1:
                        raise RuntimeError("SWE task must have one frozen preflight image")
                    image = images[0]
                timeout = settings.get("swe_agent_timeout_seconds")
                if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout < float("inf"):
                    raise RuntimeError("Explicit finite SWE agent deadline is required")
                from .scoped_gateway import target_scope
                with target_scope(run_id, role, task_id, digest) as scoped_token:
                    result = native_evaluate(row, workspace, model, artifacts / "native_swe", run_id, role,
                                             image, timeout=timeout, ledger=ledger, scoped_token=scoped_token)
        elif benchmark == "gaia2":
            if not settings.get("judge"):
                raise RuntimeError("GAIA2 requires an explicit official judge model/endpoint configuration")
            env = os.environ.copy()
            env["DEMO_FS_PATH"] = str(ROOT / "third_party/meta_are_data/gaia2_filesystem/132e26376f5e963bb59f64bcccdd02188cb08dee/demo_filesystem")
            env["PYTHONUTF8"] = "1"
            config = {"task_id": task_id, "candidate": str(workspace), "model": model, "judge": settings["judge"], "artifacts": str(artifacts / "are"), "ledger": str(ledger_path), "run_id": run_id, "role": role}
            config_path, result_path = artifacts / "are_input.json", artifacts / "are_result.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            completed = subprocess.run([str(ROOT / "third_party/Meta-ARE/.venv/Scripts/python.exe"), "-m", "experiments.autosaddler_reproduction.gaia2_adapter", "--config", str(config_path), "--result", str(result_path)], cwd=ROOT, env=env, capture_output=True, encoding="utf-8", timeout=2400)
            (artifacts / "are_stdout.log").write_text(completed.stdout, encoding="utf-8")
            (artifacts / "are_stderr.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode or not result_path.exists():
                raise RuntimeError("GAIA2 run failed; inspect ARE logs")
            result = json.loads(result_path.read_text())
        else:
            raise ValueError("Unknown benchmark")
        if not result.get("valid_evaluation"):
            raise RuntimeError("Adapter did not return a valid benchmark evaluation")
        if candidate_hash(workspace) != digest and benchmark == "gaia2":
            raise RuntimeError("Target modified its frozen harness during evaluation")
        result.update({"benchmark": benchmark, "task_id": task_id, "candidate_hash": digest, "model": model})
        result["interactions"] = agent_interactions(artifacts)
        if benchmark == "swe_bench_pro":
            prompt = "\n\n".join(str(row.get(name, "")) for name in ("problem_statement", "requirements", "interface"))
            result["interactions"].insert(0, {"task_prompt": prompt})
            result["validation_rationale"] = f"Passed {result['passed_required_tests']}/{result['required_tests']} required official tests; failed: " + ", ".join(result.get("failed_required_tests", []))
        elif benchmark == "terminal_bench_2":
            prompt = (ROOT / "third_party/terminal-bench-2" / task_id / "instruction.md").read_text(encoding="utf-8")
            result["interactions"].insert(0, {"task_prompt": prompt})
            feedback = list((artifacts / "harbor").rglob("test-stdout.txt"))
            if feedback:
                result["validation_rationale"] = feedback[0].read_text(encoding="utf-8", errors="replace")[-12000:]
        usage = {}
        for request_id, request_role, payload in case_rows():
            if request_id in previous_requests:
                continue
            counters = json.loads(payload)
            bucket = usage.setdefault(request_role, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "usage_complete": True, "calls": 0})
            bucket["calls"] += 1
            bucket["usage_complete"] = bucket["usage_complete"] and counters.get("usage_complete", False)
            for field in ("input_tokens", "output_tokens", "total_tokens"):
                bucket[field] += counters.get(field) or 0
        for bucket in usage.values():
            if not bucket["usage_complete"]:
                bucket["total_tokens"] = None
        result["usage"] = usage
        return result
    finally:
        ledger.close()
        if settings.get('fullsource_manifest') and workspace.exists():
            # Candidate Git objects and the build proof retain the exact source;
            # a second full tree per target attempt would exhaust this host.
            shutil.rmtree(workspace)


def evaluate_final(runtime, repetitions=3, n_test=None, selection_runtime=None):
    from autosaddler.v2.core.run_state import RunState
    from autosaddler.v2.core.domain import Case
    benchmarks = selected_benchmarks(runtime.config.scenario.settings)
    selection_runtime = selection_runtime or runtime
    if runtime.config.scenario.settings.get('require_complete_optimization_stages'):
        from .complete_engine import require_complete_search
        require_complete_search(selection_runtime.store,selection_runtime.config.optimization.budget.max_iterations)
    state = RunState.replay(selection_runtime.store.events())
    if not state.optimization_completed or not state.selected_candidate_id:
        raise RuntimeError("Final evaluation requires a completed optimization with a selected candidate")
    if runtime.config.scenario.settings["mode"] != "real":
        raise ValueError("Synthetic smoke candidates cannot produce a benchmark results table")
    seed = selection_runtime.scenario.harness_space.seed()
    selected = state.candidates[state.selected_candidate_id]
    if repetitions < 1 or (n_test is not None and n_test < 1):
        raise ValueError("Test repetitions and sample size must be positive")
    plan = {"benchmarks": list(benchmarks), "candidate_id":selected.candidate_id, "seed_id":seed.candidate_id,
            "repetitions":repetitions, "n_test":n_test, "model":runtime.config.scenario.settings["target_model"],
            "case_selection_seed":runtime.config.scenario.settings.get("case_selection_seed")}
    swe_paired_order = list(benchmarks) == ["swe_bench_pro"]
    if swe_paired_order:
        plan.update(order_policy="task_repetition_paired_sha256_v1", order_seed=20261106)
    if selection_runtime is not runtime:
        plan["selection_run_id"] = selection_runtime.store.run_id
    plan_path = runtime.store.run_dir / "final_evaluation_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("The final test plan is frozen; changing it requires a new experiment")
    plan_path.write_text(json.dumps(plan,indent=2),encoding="utf-8")
    labels = {"H0": (seed, "target_final_baseline"), "H1": (selected, "target_final_optimized")}
    from .case_selection import select_ids
    from .case_selection import manifest_path
    selected_test = {benchmark: select_ids(json.loads(manifest_path(runtime.config.scenario.settings, benchmark).read_text())["test"],
                     n_test, runtime.config.scenario.settings.get("case_selection_seed"), benchmark, "test")
                     for benchmark in benchmarks}
    schedule = []
    if swe_paired_order:
        for task_id in selected_test["swe_bench_pro"]:
            for repeat in range(repetitions):
                parity = hashlib.sha256(f"20261106:{task_id}:{repeat}".encode()).digest()[0] & 1
                order = ("H0", "H1") if parity == 0 else ("H1", "H0")
                for label in order:
                    schedule.append((label, "swe_bench_pro", task_id, repeat, order[0]))
    else:
        for label in ("H0", "H1"):
            for benchmark in benchmarks:
                for task_id in selected_test[benchmark]:
                    for repeat in range(repetitions):
                        schedule.append((label, benchmark, task_id, repeat, None))
    results = []
    for index, (label, benchmark, task_id, repeat, first_label) in enumerate(schedule):
        candidate, role = labels[label]
        path = runtime.store.run_dir / "final_test" / label / benchmark / hashlib.sha256(task_id.encode()).hexdigest()[:16] / str(repeat)
        result_path = path / "result.json"
        expected = {"candidate_id": candidate.candidate_id,
                    "model": runtime.config.scenario.settings["target_model"],
                    "benchmark": benchmark, "task_id": task_id,
                    "label": label, "repetition": repeat}
        if swe_paired_order:
            expected.update(execution_index=index, planned_first_label=first_label)
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if any(result.get(key) != value for key, value in expected.items()) or result.get("valid_evaluation") is not True or result.get("score") not in (0, 0.0, 1, 1.0):
                raise RuntimeError("Cached test result provenance drift")
            if benchmark == "swe_bench_pro" and result.get("synthetic") and not runtime.config.provider.type == "experiment_smoke":
                raise RuntimeError("Synthetic SWE result cannot be reused in a real experiment")
            if benchmark == "swe_bench_pro" and not result.get("synthetic"):
                if not result.get("completed_at_utc"):
                    raise RuntimeError("Cached SWE result lacks completion timestamp")
                bucket = result.get("usage", {}).get(role, {})
                if (result.get("agent_cleanup_verified") is not True or
                    result.get("generation_container_removed_before_grading") is not True or
                    not result.get("grading_image_id") or bucket.get("usage_complete") is not True or
                    bucket.get("calls", 0) < 1 or bucket.get("total_tokens") is None):
                    raise RuntimeError("Cached SWE result lacks complete grading or usage evidence")
        else:
            case = Case(f"{benchmark}:{task_id}", "test", {"benchmark": benchmark, "task_id": task_id})
            materialized = selection_runtime.scenario.harness_space.materialize(candidate, "evaluate")
            try:
                result = run_case(case, materialized.root, path, runtime.config.scenario.settings,
                                  runtime.store.run_id, role)
            finally:
                materialized.release()
            result.update(expected)
            if swe_paired_order:
                from datetime import datetime, timezone
                result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            path.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
    table = []
    usage_ledger = UsageLedger(ROOT / "results/autosaddler_reproduction/usage" / runtime.store.run_id / "requests.sqlite")
    for benchmark in benchmarks:
        before = [r["score"] for r in results if r["label"] == "H0" and r["benchmark"] == benchmark]
        after = [r["score"] for r in results if r["label"] == "H1" and r["benchmark"] == benchmark]
        usage = usage_ledger.totals(runtime.store.run_id, benchmark)
        baseline_usage = usage.get("target_final_baseline", {})
        optimized_usage = usage.get("target_final_optimized", {})
        table.append({"benchmark": benchmark, "attempts_per_harness": len(before), "H0_pass_at_1": sum(before)/len(before), "H1_pass_at_1": sum(after)/len(after), "delta_pp": 100*(sum(after)/len(after)-sum(before)/len(before)),
            "H0_input_tokens":baseline_usage.get("input_tokens"), "H0_output_tokens":baseline_usage.get("output_tokens"), "H0_total_tokens":baseline_usage.get("total_tokens"),
            "H1_input_tokens":optimized_usage.get("input_tokens"), "H1_output_tokens":optimized_usage.get("output_tokens"), "H1_total_tokens":optimized_usage.get("total_tokens"),
            "optimization_run_id":selection_runtime.store.run_id})
    with (runtime.store.run_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    # One shared optimization cost, never repeated and summed across benchmarks.
    optimizer = [e.payload["usage"] for e in selection_runtime.store.events_of_type("ModelUsageObserved") if e.payload.get("usage",{}).get("role") == "optimizer"]
    total = lambda field: sum(u.get(field) or 0 for u in optimizer)
    summary = {"optimization_run_id":selection_runtime.store.run_id, "optimizer_input_tokens":total("input_tokens"), "optimizer_output_tokens":total("output_tokens"),
               "optimizer_total_tokens":None if any(u.get("usage_incomplete") for u in optimizer) else total("input_tokens")+total("output_tokens"),
               "usage_by_role":usage_ledger.totals(runtime.store.run_id)}
    (runtime.store.run_dir / "token_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    usage_ledger.close()
    return table
