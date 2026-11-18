"""Readiness checks distinguish technical fixtures from paid benchmark results."""
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from .benchmarks import BENCHMARKS, selected_benchmarks

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).parent
RESULTS = ROOT / "results/autosaddler_reproduction"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def inspect_environment(benchmarks=BENCHMARKS):
    benchmarks = selected_benchmarks({"benchmarks": benchmarks})
    checks, notes = {}, []
    checks["python_312_plus"] = sys.version_info >= (3,12)
    for name, counts in (("gaia2", (75,65,300)), ("swe_bench_pro", (79,80,237)), ("terminal_bench_2", (30,19,40))):
        if name not in benchmarks:
            continue
        manifest = read_json(PACKAGE / "manifests" / f"{name}.json")
        checks[name + "_manifest"] = bool(manifest) and tuple(len(manifest[s]) for s in ("train","development","test")) == counts
        if manifest and manifest.get("split_origin"):
            notes.append(name + ": " + manifest["split_origin"])
    if "gaia2" in benchmarks:
        checks["gaia2_payloads"] = len(list((ROOT / "third_party/experiment_data/gaia2").rglob("[0-9]*.json"))) == 440
    if "swe_bench_pro" in benchmarks:
        checks["swe_v1_payload"] = (ROOT / "third_party/experiment_data/swe_bench_pro_v1.jsonl").is_file()
    versions = {name:importlib.metadata.version(name) for name in ("autosaddler","harbor","datasets","pytest","docker")}
    checks["pinned_core_versions"] = versions["autosaddler"] == "0.2.0" and versions["harbor"] == "0.25.0"
    try:
        docker = subprocess.run(["docker","info","--format","{{.NCPU}} {{.MemTotal}}"],capture_output=True,text=True,timeout=15)
        cpu, memory = map(int,docker.stdout.strip().split())
        checks["docker_resources"] = docker.returncode == 0 and cpu >= 4 and memory >= 8*1024**3
        image = subprocess.run(["docker","image","inspect","autoharness-opencode:1.19.5","--format","{{.Id}}"],capture_output=True,text=True,timeout=10)
        checks["opencode_container"] = image.returncode == 0
        image_id = image.stdout.strip()
    except (OSError, ValueError, subprocess.TimeoutExpired):
        checks["docker_resources"] = checks["opencode_container"] = False
        image_id = None
    for name, path, field in (
        ("native_gaia2_oracle", "infrastructure/gaia2_oracle/result.json", "native_oracle_replay"),
        ("gaia2_opencode_events", "infrastructure/gaia2_event_smoke/result.json", "passed"),
        ("swe_official_positive_negative", "infrastructure/swe_v1_oracle/result.json", "passed"),
        ("opencode_acp_wire_and_usage", "opencode_mock_wire/result.json", "passed"),
        ("weak_model_acp", "acp_model_smoke/result.json", "passed"),
        ("autosaddler_git_and_resume", "engine_smoke_result.json", "passed"),
    ):
        if "gaia2" in name and "gaia2" not in benchmarks:
            continue
        if name.startswith("swe_") and "swe_bench_pro" not in benchmarks:
            continue
        proof = read_json(RESULTS / path)
        checks[name] = bool(proof and proof.get(field))
    oracle = read_json(RESULTS / "infrastructure/tb2_setup_oracle/result.json")
    checks["terminal_bench_native_oracle"] = bool(oracle and not any((r.get("exception_info") for r in [read_json(p) or {} for p in (RESULTS / "infrastructure/tb2_setup_oracle").glob("*/result.json")])) )
    installs = list((RESULTS / "infrastructure/opencode_install_v2/trial").glob("*/result.json"))
    checks["shared_opencode_harbor_install"] = len(installs) == 1 and not read_json(installs[0]).get("exception_info")
    if "terminal_bench_2" not in benchmarks:
        checks.pop("terminal_bench_native_oracle")
        checks.pop("shared_opencode_harbor_install")
    junit = RESULTS / "unit-tests.xml"
    checks["unit_tests"] = False
    if junit.exists():
        suites = ET.parse(junit).getroot().findall("testsuite")
        checks["unit_tests"] = bool(suites) and all(int(s.get("failures",0)) == 0 and int(s.get("errors",0)) == 0 for s in suites)
    return {"benchmarks":list(benchmarks), "checked_at_utc":datetime.now(timezone.utc).isoformat(), "environment_ready_for_model_binding":all(checks.values()),
        "checks":checks, "versions":versions, "optimizer_image_id":image_id,
        "pending_model_binding":["OpenCode provider credentials", "Frozen optimizer and target model IDs", "Train-only cost/quota calibration and final experiment-size freeze"] + (["Explicit GAIA2 judge model/endpoint and its usage budget"] if "gaia2" in benchmarks else []),
        "methodology_notes":["Official AutoSaddler V2 engine, with custom joint-benchmark sampling/ranking; not the paper's V1 executable",
            "One shared H0 and one shared H1; models and adapters fixed for the comparison",
            "OpenCode title/summary and subagent tasks disabled; versioned Go native compaction is separately tagged and charged to its parent role within the existing campaign budget",
            "Only representative environment images are warmed; other task images download before their agent runs",
            "Pinned 440 GAIA2 scenario payloads contain no nonempty attachment fields; unsupported future attachment variants fail explicitly",
            "Fixtures/oracles are infrastructure evidence, not measured H0/H1 benchmark performance", *notes],
        "paid_experiments_started":False}


def check_real_configuration(runtime):
    settings = runtime.config.scenario.settings
    if settings.get("launch_ready") is False:
        raise RuntimeError("Draft launch is blocked: verify Go Plus authentication, quota control and calibration first")
    if settings.get("require_go_guard") and not os.environ.get("GO_GATEWAY_URL"):
        raise RuntimeError("Start the metered Go launcher; direct unguarded execution is forbidden")
    benchmarks = selected_benchmarks(settings)
    report = inspect_environment(benchmarks)
    missing = [k for k,v in report["checks"].items() if not v]
    if missing:
        raise RuntimeError("Environment checks failed: " + ", ".join(missing))
    if settings.get("mode") != "real":
        raise ValueError("Real execution requires mode=real")
    if not settings.get("target_model") or not runtime.config.provider.settings.get("model"):
        raise ValueError("Set explicit target and optimizer model IDs")
    if "gaia2" in benchmarks and not settings.get("judge"):
        raise ValueError("Configure the GAIA2 judge model/provider/endpoint before execution")
    per_benchmark = int(settings.get("batch_per_benchmark", 1))
    if runtime.config.optimization.task_selection.batch_size != len(benchmarks) * per_benchmark:
        raise ValueError("Batch size must match selected benchmarks times batch_per_benchmark")
    gateway_models = [settings["target_model"], runtime.config.provider.settings["model"]]
    if any(model.startswith("codex-acp/") for model in gateway_models):
        if not all(model == "codex-acp/gpt-6-luna" for model in gateway_models):
            raise ValueError("ACP experiment must freeze the same declared transport/model for target and optimizer")
        if not os.environ.get("ACP_GATEWAY_TOKEN") or not os.environ.get("ACP_GATEWAY_URL"):
            raise ValueError("Start the local ACP model gateway before real execution")
    elif not os.environ.get("OPENCODE_GO_API_KEY"):
        raise ValueError("OPENCODE_GO_API_KEY is missing; no model calls were made")


def main():
    report = inspect_environment()
    RESULTS.mkdir(parents=True,exist_ok=True)
    (RESULTS / "readiness.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    return 0 if report["environment_ready_for_model_binding"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
