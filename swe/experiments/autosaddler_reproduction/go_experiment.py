"""Bounded Terminal-Bench Go Plus calibration and frozen experiment launcher."""
import argparse
import csv
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import traceback
import yaml

from .go_gateway import GoGateway, save, provider_config, ROOT
from .acp_gateway import context_headers

PACKAGE = Path(__file__).parent


def check_preflight(args):
    """Fail before inference unless the exact selected environments were installed."""
    import yaml
    import subprocess
    from .engine import prepare_seed
    from .case_selection import select_ids
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = config["scenario"]["settings"]
    if settings.get("context_policy_version"):
        from .context_policy import VERSION, PLUGIN
        proof = json.loads((ROOT / "results/autosaddler_reproduction/context_policy_fixture/latest.json").read_text())
        if settings["context_policy_version"] != VERSION or not proof.get("passed") or proof.get("paid_model_requests") != 0 or proof.get("policy_plugin_sha256") != hashlib.sha256(PLUGIN.encode()).hexdigest():
            raise RuntimeError("Native context policy offline proof is missing or stale")
        if settings.get("request_guard_version", 1) >= 2 and (proof.get("request_guard_version") != settings["request_guard_version"] or not proof.get("native_media_over_210kb_verified")):
            raise RuntimeError("Native media request guard proof is missing or stale")
        if settings.get("request_guard_version", 1) >= 3 and not proof.get("long_native_response_compaction_verified"):
            raise RuntimeError("Long native response compaction proof is missing")
        if settings.get("optimizer_scratch_access_version", 1) >= 2 and not proof.get("direct_tmp_native_read_verified"):
            raise RuntimeError("Native direct scratch read proof is missing")
        if settings.get("stream_transport_version") and proof.get("stream_transport_version") != settings["stream_transport_version"]:
            raise RuntimeError("Native streaming transport proof is missing or stale")
        if settings.get("request_timeout_version") and proof.get("request_timeout_version") != settings["request_timeout_version"]:
            raise RuntimeError("Native transport timeout proof is missing or stale")
    if settings.get("require_cleanup_proof"):
        lifecycle = json.loads((ROOT / "results/autosaddler_reproduction/terminal_cleanup_proof.json").read_text())
        if not lifecycle.get("passed") or lifecycle.get("model_calls") != 0:
            raise RuntimeError("Process cleanup regression proof is missing")
    if settings.get('require_complete_optimization_stages'):
        proof=json.loads((ROOT/'results/autosaddler_reproduction/full_protocol_preparation/fixture/latest.json').read_text())
        if not proof.get('passed') or proof.get('external_model_calls')!=0 or not proof.get('structured_output_repair_verified'):
            raise RuntimeError('Complete native optimizer lifecycle proof is required')
        if any(hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()!=digest for name,digest in proof['source_sha256'].items()):
            raise RuntimeError('Complete optimizer lifecycle proof is stale')
    if settings["benchmarks"] != ["terminal_bench_2"]:
        raise RuntimeError("Go launcher is frozen to Terminal-Bench only")
    manifest = json.loads((PACKAGE / "manifests/terminal_bench_2.json").read_text())
    selected = {split:select_ids(manifest[split], n, settings["case_selection_seed"], "terminal_bench_2", split)
        for split,n in (("train", settings["train_per_benchmark"]), ("development", settings["development_per_benchmark"]), ("test", args.n_test))}
    preflight = (ROOT / settings.get('environment_preflight_directory', 'results/autosaddler_reproduction/terminal_go_preflight')).resolve()
    if not preflight.is_relative_to(ROOT.resolve()):
        raise RuntimeError('Environment preflight path escapes workspace')
    proof = json.loads((preflight / "completed.json").read_text(encoding="utf-8"))
    _, revision = prepare_seed()
    # Environment setup order is independent of the frozen evaluation order.
    normalized = lambda values: {key:sorted(tasks) for key,tasks in values.items()}
    if normalized(proof["selected"]) != normalized(selected) or proof["seed_revision"] != revision:
        raise RuntimeError("Environment preflight does not match the frozen experiment")
    for tasks in selected.values():
        for task_id in tasks:
            installed = json.loads((preflight / task_id / "installed.json").read_text(encoding="utf-8"))
            if not installed.get("installed") or installed["seed_revision"] != revision or installed["inference_calls"] != 0:
                raise RuntimeError("Missing environment-only proof: " + task_id)
            image = proof["images"][task_id]
            current = subprocess.check_output(["docker", "image", "inspect", image["tag"], "--format", "{{.Id}}"], encoding="utf-8", timeout=30).strip()
            if current != image["id"]:
                raise RuntimeError("Prepared task image changed: " + task_id)
    return proof


def wire(gateway, artifacts):
    from .opencode_runtime import OpenCodeSession
    from .accounting import UsageLedger
    workspace = ROOT / ".test_tmp/go-wire" / artifacts.name
    workspace.mkdir(parents=True, exist_ok=True)
    for name in ("instructions.md", "opencode.json"):
        shutil.copy2(PACKAGE / "harness_seed" / name, workspace / name)
    (workspace / "fixture.txt").write_text("left=17\nright=25\n")
    model = "opencode-go/kimi-k2.7-code"
    with OpenCodeSession(workspace, artifacts / "wire", model, container=True,
        overlay={"provider":provider_config(model, context_headers(artifacts.name, "optimizer", "fixture", "wire"))}) as session:
        result = session.prompt('Read /workspace/fixture.txt, add left and right, write the sum to /workspace/answer.txt. Verify it with a tool and reply {"answer":42}.', timeout=240)
        ledger = UsageLedger(artifacts / "wire_usage.sqlite")
        try:
            session.export_usage(ledger, artifacts.name, "optimizer")
            usage = ledger.totals(artifacts.name)
        finally:
            ledger.close()
    if not (workspace / "answer.txt").exists() or (workspace / "answer.txt").read_text().strip() != "42":
        raise RuntimeError("Optimizer native tools wire check failed")
    save(artifacts / "wire_result.json", {"passed":True, "usage":usage})


def calibrate(args, gateway, artifacts):
    from autosaddler.v2.core.domain import Case
    from .engine import prepare_seed
    from .runner import run_case
    repo, revision = prepare_seed()
    settings = {"benchmarks":["terminal_bench_2"], "target_model":"opencode-go/qwen3.7-plus"}
    path = artifacts / "calibration_result.json"
    if not path.exists():
        task_id = "log-summary-date-ranges"
        print("Train calibration: " + task_id, flush=True)
        attempt = len(list((artifacts / "train").glob("attempt-*"))) + 1
        result = run_case(Case("terminal_bench_2:"+task_id, "train", {"benchmark":"terminal_bench_2", "task_id":task_id}),
            repo, artifacts / "train" / ("attempt-"+str(attempt)), settings, args.run_id, "target_search")
        result["seed_revision"] = revision
        save(path, result)
        print("Train calibration complete; score=" + str(result["score"]), flush=True)
    if not (artifacts / "wire_result.json").exists():
        wire(gateway, artifacts)


def load_selection(config_path, run_id):
    """Read a completed selection without reinitializing historical fingerprints."""
    from types import SimpleNamespace
    from autosaddler.v2.config.models import RunConfig
    from autosaddler.v2.storage.local import LocalRunStore
    from .engine import build_scenario
    config = RunConfig.load(config_path.resolve())
    run_dir = config.storage.run_root / run_id
    if not (run_dir / "events.jsonl").is_file():
        raise RuntimeError("Selection source journal is missing")
    store = LocalRunStore(run_dir=run_dir, run_id=run_id)
    store.validate_integrity()
    scenario = build_scenario(settings=config.scenario.settings, base_dir=config_path.parent,
        run_dir=run_dir, store=store, ledger=None)
    return SimpleNamespace(config=config, store=store, scenario=scenario)


def run(args, gateway, artifacts):
    from .engine import build
    from .doctor import check_real_configuration
    from .runner import evaluate_final
    from .case_selection import select_ids
    runtime = build(args.config.resolve(), args.run_id)
    check_real_configuration(runtime)
    manifest = json.loads((PACKAGE / "manifests/terminal_bench_2.json").read_text())
    plan = {"train":[c.case_id for c in runtime.scenario.train_cases], "development":[c.case_id for c in runtime.scenario.development_cases],
        "test":select_ids(manifest["test"], args.n_test, 42, "terminal_bench_2", "test"),
        "test_repetitions":getattr(args,'test_repetitions',1), "models":{"target":runtime.config.scenario.settings["target_model"], "optimizer":runtime.config.provider.settings["model"]},
        "max_iterations":runtime.config.optimization.budget.max_iterations, "allowance_limit":args.allowance_limit, "request_limit":args.request_limit,
        "role_allowance_limits":gateway.role_limits, "role_request_limits":gateway.role_request_limits,
        "budget_prior_runs":runtime.config.scenario.settings.get("budget_prior_runs", [])}
    from .context_policy import VERSION, INPUT_LIMIT, TOOL_OUTPUT_BYTES, PLUGIN
    from .go_gateway import TEXT_REQUEST_BYTES, MEDIA_REQUEST_BYTES, REQUEST_GUARD_VERSION, RESERVATION_VERSION, BILLED_OUTPUT_RESERVE_TOKENS, STREAM_TRANSPORT_VERSION
    plan["context_policy"] = {"version":VERSION, "native_input_limit":INPUT_LIMIT,
        "reserved_tokens":8192, "compaction_max_output_tokens":4096, "tool_output_excerpt_trigger_bytes":TOOL_OUTPUT_BYTES,
        "plugin_sha256":hashlib.sha256(PLUGIN.encode()).hexdigest(), "compaction_model":"same as parent session",
        "accounting":"purpose=compaction, included in parent role and campaign limits", "request_byte_limit":TEXT_REQUEST_BYTES,
        "media_transport_byte_limit":MEDIA_REQUEST_BYTES, "request_guard_version":REQUEST_GUARD_VERSION,
        "media_reservation":"full serialized bytes, including binary attachments"}
    plan["reservation_policy"] = {"version":RESERVATION_VERSION, "billed_output_reserve_tokens":BILLED_OUTPUT_RESERVE_TOKENS,
        "request_output_limit":8192, "native_payload_unchanged":True}
    plan["stream_transport"] = {"version":STREAM_TRANSPORT_VERSION, "native_sse_forwarded_incrementally":True,
        "budget_admission_atomic":True, "upstream_io_outside_budget_lock":True, "disconnected_callers_rejected":True,
        "terminal_frame_requires_verified_billing":True}
    if runtime.config.scenario.settings.get('require_complete_optimization_stages'):
        from .go_gateway import DISCONNECT_WATCH_VERSION
        plan['stream_transport']['disconnect_watch_version']=DISCONNECT_WATCH_VERSION
    selection_runtime = None
    from .go_gateway import REQUEST_TIMEOUT_VERSION, REQUEST_TIMEOUT_SECONDS
    plan["request_timeout_policy"] = {"version":REQUEST_TIMEOUT_VERSION, "previous_seconds":600,
        "sdk_and_http_read_seconds":REQUEST_TIMEOUT_SECONDS, "official_task_deadlines_unchanged":True}
    selection_run_id = getattr(args, "selection_run_id", None)
    if selection_run_id:
        if selection_run_id == args.run_id or selection_run_id not in plan["budget_prior_runs"]:
            raise RuntimeError("Evaluation-only selection must be a distinct accounted prior run")
        source = artifacts.parent / selection_run_id
        if json.loads((source / "status.json").read_text())["status"] not in {"stopped", "complete"}:
            raise RuntimeError("Selection source must not be running")
        selection_config = args.selection_config.resolve()
        selection_runtime = load_selection(selection_config, selection_run_id)
        from autosaddler.v2.core.run_state import RunState
        selection = RunState.replay(selection_runtime.store.events())
        if not selection.optimization_completed or not selection.selected_candidate_id:
            raise RuntimeError("Selection source optimization is incomplete")
        for key in set(runtime.config.scenario.settings) | set(selection_runtime.config.scenario.settings):
            if key in {"budget_prior_runs", "reservation_version", "stream_transport_version", "request_timeout_version", "request_guard_version"}:
                continue
            if runtime.config.scenario.settings.get(key) != selection_runtime.config.scenario.settings.get(key):
                raise RuntimeError("Evaluation-only source policy drift: " + key)
        plan["selection_source"] = {"run_id":selection_run_id, "candidate_id":selection.selected_candidate_id,
            "seed_id":selection_runtime.scenario.harness_space.seed().candidate_id,
            "events_sha256":hashlib.sha256((selection_runtime.store.run_dir / "events.jsonl").read_bytes()).hexdigest(),
            "config_sha256":hashlib.sha256(selection_config.read_bytes()).hexdigest(), "new_optimization":False,
            "prior_scores_imported":False}
    path = artifacts / "frozen_plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise RuntimeError("Cannot change frozen experiment plan")
    save(path, plan)
    if selection_runtime is None:
        result = runtime.engine.run()
        save(artifacts / "optimization_result.json", {"selected_candidate_id":result.selected_candidate_id, "iterations":result.iterations, "development_score":result.development_score})
    else:
        save(artifacts / "selection_provenance.json", plan["selection_source"])
    comparison = evaluate_final(runtime, repetitions=plan['test_repetitions'], n_test=args.n_test, selection_runtime=selection_runtime)
    if gateway.halted:
        raise RuntimeError("Gateway transport/billing failure invalidates final report: " + gateway.halted)
    # Native CLI logs can miss usage for a request interrupted at timeout. The
    # proxy's provider counters include every billed attempt, including failures.
    provider_usage = gateway.accounting_summary()
    save(artifacts / "campaign_accounting.json", provider_usage)
    save(artifacts / "native_comparison.json", comparison)
    for row in comparison:
        for label, role in (("H0", "target_final_baseline"), ("H1", "target_final_optimized")):
            usage = provider_usage["roles"].get(role, {})
            for field in ("input_tokens", "output_tokens"):
                row[label+"_"+field] = usage.get(field)
            row[label+"_total_tokens"] = usage.get("tokens")
        row["token_source"] = "Go provider usage, including all charged attempts"
    save(artifacts / "comparison.json", comparison)
    with (artifacts / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)


def optimizer_check(args, artifacts):
    """Real optimizer protocol with synthetic target scores, never benchmark data."""
    import yaml
    from .engine import build
    config = yaml.safe_load((PACKAGE / "configs/terminal_go_plus.draft.yaml").read_text())
    config["scenario"]["settings"].update(mode="smoke", capability_phase_iterations=1, batch_per_benchmark=1)
    config["optimization"]["task_selection"]["batch_size"] = 1
    config["optimization"]["budget"] = {"max_iterations":1, "max_rollouts":4}
    config["storage"]["run_root"] = str(artifacts / "synthetic_target")
    path = artifacts / "optimizer_check.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    runtime = build(path, args.run_id)
    result = runtime.engine.run()
    save(artifacts / "optimizer_check.json", {"completed":True, "real_optimizer":True, "synthetic_target":True,
        "iterations":result.iterations, "selected_candidate_id":result.selected_candidate_id,
        "note":"Protocol and Git mutation check only, not benchmark performance"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["calibrate", "optimizer-check", "run"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--allowance-limit", type=float, default=5)
    parser.add_argument("--request-limit", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=5)
    parser.add_argument('--test-repetitions',type=int,default=1)
    parser.add_argument("--selection-run-id")
    parser.add_argument("--selection-config", type=Path)
    args = parser.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("Unsafe run ID")
    if args.phase == "run" and args.config is None:
        parser.error("Run requires frozen --config")
    if bool(args.selection_run_id) != bool(args.selection_config):
        parser.error("Evaluation-only mode requires both selection source arguments")
    if args.selection_run_id and (args.phase != "run" or not args.selection_run_id.replace("-", "").replace("_", "").isalnum()):
        parser.error("Invalid evaluation-only selection source")
    artifacts = ROOT / "results/autosaddler_reproduction/go_experiments" / args.run_id
    status = {"status":"running", "phase":args.phase, "pid":os.getpid(), "started":datetime.now(timezone.utc).isoformat()}
    save(artifacts / "status.json", status)
    limits = None if args.phase != "run" else {"optimizer":args.allowance_limit*0.2, "target_search":args.allowance_limit*0.4,
        "target_final_baseline":args.allowance_limit*0.2, "target_final_optimized":args.allowance_limit*0.2}
    request_limits = None if args.phase != "run" else {"optimizer":int(args.request_limit*0.1), "target_search":int(args.request_limit*0.5),
        "target_final_baseline":int(args.request_limit*0.2), "target_final_optimized":int(args.request_limit*0.2)}
    gateway = None
    try:
        prior_journals = []
        if args.phase == "run":
            save(artifacts / "environment_preflight.json", check_preflight(args))
            config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            if config['scenario']['settings'].get('require_complete_optimization_stages'):
                limits=config['scenario']['settings'].get('role_allowance_limits',limits)
                request_limits=config['scenario']['settings'].get('role_request_limits',request_limits)
            for prior_id in config["scenario"]["settings"].get("budget_prior_runs", []):
                if not prior_id.replace("-", "").replace("_", "").isalnum() or prior_id == args.run_id:
                    raise RuntimeError("Invalid prior budget run ID")
                prior = artifacts.parent / prior_id
                if json.loads((prior / "status.json").read_text())["status"] not in {"stopped", "complete"}:
                    raise RuntimeError("Prior budget run must have stopped")
                prior_journals.append(prior / "gateway/requests.jsonl")
        with GoGateway(artifacts / "gateway", args.allowance_limit, args.request_limit, role_limits=limits, role_request_limits=request_limits, prior_journals=prior_journals, require_context_policy=args.phase == "run") as gateway:
            if args.phase == "calibrate":
                calibrate(args, gateway, artifacts)
            elif args.phase == "optimizer-check":
                optimizer_check(args, artifacts)
            else:
                run(args, gateway, artifacts)
            status.update(status="complete", allowance=gateway.spent())
    except Exception as exc:
        status.update(status="stopped", error=str(exc))
        save(artifacts / "failure.json", {"traceback":traceback.format_exc()})
    finally:
        if gateway is not None:
            save(artifacts / "campaign_accounting.json", gateway.accounting_summary())
    status["updated"] = datetime.now(timezone.utc).isoformat()
    save(artifacts / "status.json", status)
    print(json.dumps(status, indent=2), flush=True)
    return 0 if status["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
