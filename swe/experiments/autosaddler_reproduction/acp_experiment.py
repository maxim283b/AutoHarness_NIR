"""Run the shared OpenCode experiment through ACP with a weekly quota reserve.

Calibration uses training tasks only and saves each result immediately. The run
phase consumes an explicit frozen YAML plan and preserves native engine resume.
No API key fallback, top-up or automatic quota reset is performed.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import os
import subprocess

from .acp_gateway import ACPGateway, MODEL

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).parent
BENCHMARKS = ("gaia2", "swe_bench_pro", "terminal_bench_2")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def calibrate(args, gateway, artifacts):
    from autosaddler.v2.core.domain import Case
    from .runner import run_case
    from .engine import prepare_seed
    candidate, revision = prepare_seed()
    selected = {name: json.loads((PACKAGE / "manifests" / (name + ".json")).read_text())["train"][0] for name in BENCHMARKS}
    # Already provisioned representative training environment; selected for cost,
    # never based on held-out outcomes. Other train IDs remain available to search.
    selected["terminal_bench_2"] = "log-summary-date-ranges"
    settings = {"target_model": "codex-acp/" + MODEL,
        "judge": {"model_name": MODEL, "provider": "openai", "endpoint": "local-acp-gateway"}}
    outcomes = {}
    names = [args.benchmark] if args.benchmark else BENCHMARKS
    for benchmark in names:
        task_id = selected[benchmark]
        target = artifacts / "calibration" / benchmark
        result_path = target / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("seed_revision") != revision:
                raise ValueError("Calibration seed changed; choose a new run ID")
        else:
            attempts = len(list(target.glob("attempt-*"))) + 1
            started = time.monotonic()
            print("Calibration start: " + benchmark + " / " + task_id, flush=True)
            result = run_case(Case(benchmark+":"+task_id, "train", {"benchmark": benchmark, "task_id": task_id}),
                candidate, target / ("attempt-" + str(attempts)), settings, args.run_id, "target_search")
            result.update({"elapsed_sec": round(time.monotonic()-started, 2), "seed_revision": revision,
                "quota_remaining_after": gateway.read_quota(), "gateway_completions_so_far": gateway.requests})
            save(result_path, result)
        outcomes[benchmark] = result
        print("Calibration complete: " + benchmark + "; score=" + str(result["score"]), flush=True)
        save(artifacts / "calibration_summary.json", outcomes)
    probe = artifacts / "judge_route_probe.json"
    if not probe.exists():
        checked = subprocess.run([str(ROOT / "third_party/Meta-ARE/.venv/Scripts/python.exe"), "-m", "experiments.autosaddler_reproduction.judge_route_probe",
            "--run-id", args.run_id, "--result", str(probe)], cwd=ROOT, env=os.environ.copy(), capture_output=True, encoding="utf-8", errors="replace", timeout=240)
        (artifacts / "judge_route_probe.log").write_text(checked.stdout + checked.stderr, encoding="utf-8")
        if checked.returncode or not probe.exists():
            raise RuntimeError("ARE judge model route check failed; inspect saved probe log")
    return outcomes


def experiment(args, gateway, artifacts):
    if not args.config:
        raise ValueError("Run phase requires a frozen --config")
    from .engine import build
    from .doctor import check_real_configuration
    from .runner import evaluate_final
    runtime = build(args.config.resolve(), args.run_id)
    check_real_configuration(runtime)
    from .case_selection import select_ids
    from .benchmarks import selected_benchmarks
    plan = {"run_id": args.run_id, "config": str(args.config), "minimum_weekly_remaining": args.minimum_weekly_remaining,
        "optimizer_model": runtime.config.provider.settings["model"], "target_model": runtime.config.scenario.settings["target_model"],
        "judge": runtime.config.scenario.settings["judge"], "max_iterations": runtime.config.optimization.budget.max_iterations,
        "train_case_ids": [c.case_id for c in runtime.scenario.train_cases], "development_case_ids": [c.case_id for c in runtime.scenario.development_cases],
        "test_repetitions": args.test_repetitions, "test_task_ids": {name:select_ids(json.loads((PACKAGE/"manifests"/(name+".json")).read_text())["test"], args.n_test,
            runtime.config.scenario.settings.get("case_selection_seed"), name, "test") for name in selected_benchmarks(runtime.config.scenario.settings)}}
    plan_path = artifacts / "frozen_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Cannot change frozen plan within the same experiment")
    save(plan_path, plan)
    result = runtime.engine.run()
    print(str(result), flush=True)
    save(artifacts / "optimization_result.json", {"run_id": result.run_id, "selected_candidate_id": result.selected_candidate_id,
        "iterations": result.iterations, "development_score": result.development_score})
    comparison = evaluate_final(runtime, args.test_repetitions, args.n_test)
    save(artifacts / "comparison.json", comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["calibrate", "run"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--minimum-weekly-remaining", type=int, default=50)
    parser.add_argument("--max-requests", type=int, default=4000)
    parser.add_argument("--max-tokens", type=int, default=150000000)
    parser.add_argument("--test-repetitions", type=int, default=3)
    parser.add_argument("--n-test", type=int, default=3)
    args = parser.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("Invalid run ID")
    artifacts = ROOT / "results/autosaddler_reproduction/acp_experiments" / args.run_id
    invocation = artifacts / "invocations" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    status = {"run_id": args.run_id, "pid": os.getpid(), "phase": args.phase, "minimum_weekly_remaining": args.minimum_weekly_remaining,
        "model": MODEL, "backend": "Codex ACP typed action-selection adapter", "target_agent": "OpenCode 1.19.5",
        "judge_differs_from_paper": True, "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()}
    save(artifacts / "status.json", status)
    try:
        with ACPGateway(invocation / "gateway", args.max_requests, args.max_tokens, minimum_weekly_remaining=args.minimum_weekly_remaining) as gateway:
            status["initial_weekly_remaining"] = gateway.read_quota()
            save(artifacts / "status.json", status)
            if args.phase == "calibrate":
                calibrate(args, gateway, artifacts)
            else:
                experiment(args, gateway, artifacts)
            status.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "final_weekly_remaining": gateway.read_quota(), "completions": gateway.requests, "tokens": gateway.total_tokens})
            status["cumulative_transport_tokens"] = sum(json.loads(path.read_text())["total_tokens"] for path in (artifacts / "invocations").rglob("status.json"))
    except Exception as exc:
        status.update({"status": "stopped", "error_type": type(exc).__name__, "error": str(exc), "stopped_at_utc": datetime.now(timezone.utc).isoformat()})
        save(invocation / "failure.json", {"error": str(exc), "traceback": traceback.format_exc()})
        save(artifacts / "status.json", status)
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
        return 1
    save(artifacts / "status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
