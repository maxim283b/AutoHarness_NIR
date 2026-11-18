"""Bounded real-model checks using signed-in Codex ACP, separate from H0/H1.

ACP reports aggregate turn usage, not native OpenCode request usage. Preserve
that distinction in both the ledger and the human-readable report.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from .accounting import UsageLedger
from .acp_client import ACPClient

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).parent
MODEL = "gpt-6-luna"


def normalize_acp(result):
    raw = result.get("usage") or {}
    inputs, outputs = raw.get("inputTokens"), raw.get("outputTokens")
    complete = all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (inputs, outputs))
    if not complete:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None, "usage_complete": False, "raw": raw}
    read, write = raw.get("cachedReadTokens", 0), raw.get("cachedWriteTokens", 0)
    reasoning = raw.get("thoughtTokens", 0)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (read, write, reasoning)) or reasoning > outputs:
        raise ValueError("Invalid ACP cache/reasoning counters")
    # This ACP adapter reports uncached input separately from cache reads/writes.
    input_total = inputs + read + write
    total = input_total + outputs
    if raw.get("totalTokens", total) != total:
        raise ValueError("Inconsistent ACP turn counters")
    return {"input_tokens": input_total, "uncached_input_tokens": inputs, "output_tokens": outputs, "total_tokens": total,
        "cached_input_tokens": read, "cache_write_tokens": write,
        "reasoning_tokens": reasoning, "usage_complete": True, "raw": raw}


def allow_task_mcp_once(params, events):
    if not params.get("_meta", {}).get("is_mcp_tool_approval"):
        return None
    call_id = params.get("toolCall", {}).get("toolCallId")
    calls = [event.get("params", {}).get("update", {}) for event in events]
    matching = [call for call in calls if call.get("toolCallId") == call_id and call.get("rawInput", {}).get("server") == "acp_smoke_task"]
    if not matching:
        return None
    return next((option["optionId"] for option in params.get("options", []) if option.get("kind") == "allow_once"), None)


def task_server(artifacts, kind, container=None, cwd=None):
    args = ["-m", "experiments.autosaddler_reproduction.acp_smoke_mcp", "--kind", kind, "--audit", str(artifacts / "tool_calls.jsonl")]
    if container:
        args += ["--container", container]
    if cwd:
        args += ["--cwd", cwd]
    return {"name": "acp_smoke_task", "command": str(ROOT / "third_party/Meta-ARE/.venv/Scripts/python.exe"),
        "args": args, "env": [{"name": "PYTHONPATH", "value": str(ROOT)}, {"name": "PYTHONUTF8", "value": "1"}]}


def run_acp(prompt, artifacts, run_id, benchmark, server=None, timeout=300):
    artifacts.mkdir(parents=True, exist_ok=True)
    workspace = ROOT / ".test_tmp/acp_checks" / run_id / benchmark
    workspace.mkdir(parents=True, exist_ok=True)
    node, codex = shutil.which("node"), shutil.which("codex")
    if not node or not codex:
        raise RuntimeError("Authenticated Codex and Node are required")
    config = {"model": MODEL, "model_reasoning_effort": "low", "approval_policy": "never", "sandbox_mode": "read-only",
        "web_search": "disabled", "developer_instructions": "This is a bounded infrastructure check. Use only the client-provided acp_smoke_task MCP tools for task operations. Do not use host shell, host filesystem, browser, other MCP servers, skills, subagents or external services. Do not inspect reference solutions or evaluation artifacts. Stop when the task is complete."}
    env = {**os.environ, "CODEX_PATH": codex, "INITIAL_AGENT_MODE": "read-only", "CODEX_CONFIG": json.dumps(config)}
    # This check deliberately uses the existing ChatGPT login, without API-key fallback.
    for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "DEFAULT_AUTH_REQUEST", "MODEL_PROVIDER"):
        env.pop(name, None)
    adapter = ROOT / "third_party/codex-acp/node_modules/@agentclientprotocol/codex-acp/dist/index.js"
    started = time.monotonic()
    session_id = "uninitialized-" + uuid.uuid4().hex
    ledger = UsageLedger(artifacts.parent / "usage.sqlite")
    usage = normalize_acp({})
    try:
        with ACPClient([node, str(adapter)], workspace, artifacts, env, permission_handler=allow_task_mcp_once) as client:
            capabilities = client.initialize()
            session = client.new_session([server] if server else [])
            session_id = session["sessionId"]
            (artifacts / "capabilities.json").write_text(json.dumps({"initialize": capabilities, "session": session}, indent=2), encoding="utf-8")
            model_config = next((c for c in session.get("configOptions", []) if c.get("category") == "model" or c.get("id") == "model"), None)
            if model_config:
                client.request("session/set_config_option", {"sessionId": session_id, "configId": model_config["id"], "value": MODEL})
            else:
                client.request("session/set_model", {"sessionId": session_id, "modelId": MODEL + "[low]"})
            answer = client.prompt(session_id, prompt, timeout=timeout)
            (artifacts / "prompt_result.json").write_text(json.dumps(answer["result"], indent=2), encoding="utf-8")
            (artifacts / "answer.txt").write_text(answer["text"], encoding="utf-8")
            usage = normalize_acp(answer["result"])
            report = {"agent": "codex-acp", "model": MODEL, "session_id": session_id, "stop_reason": answer["result"].get("stopReason"),
                "usage": usage, "usage_granularity": "ACP turn aggregate, not individual model requests", "elapsed_sec": round(time.monotonic()-started, 2),
                "uses_signed_in_codex_allowance": True, "is_opencode_benchmark_result": False}
            (artifacts / "acp_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if not usage["usage_complete"]:
                raise RuntimeError("ACP response has no complete token counters")
            if report["stop_reason"] != "end_turn":
                raise RuntimeError("ACP task did not finish normally")
            return {**report, "text": answer["text"]}
    finally:
        ledger.record("acp:" + session_id + ":turn:1", run_id, "target_search", usage,
            benchmark=benchmark, model=MODEL, accounting_unit="ACP turn aggregate", infrastructure_only=True)
        ledger.close()


def protocol(artifacts, run_id):
    result = run_acp('ACP protocol smoke: add the integers from this client-provided fixture: left=17, right=25. Reply exactly {"answer":42}. Do not use tools.', artifacts, run_id, "protocol", timeout=120)
    parsed = json.loads(result.pop("text").removeprefix("```json").removesuffix("```").strip())
    return {**result, "passed": parsed == {"answer": 42}, "response": parsed, "official_score": False}


def gaia(artifacts, run_id):
    title = "ACP_INFRASTRUCTURE_" + run_id
    prompt = f"This is a calendar-tool integration check in an isolated GAIA2 training scenario, not the original benchmark task. Use only acp_smoke_task MCP tools. First search calendar for title {title!r}; it should not exist. Add exactly one event titled {title!r}, start_datetime='2024-10-16 10:00:00', end_datetime='2024-10-16 10:30:00'. Search it again and verify the title and dates in the actual tool response. Send a short completion message via AgentUserInterface. Then finish without other tools."
    result = run_acp(prompt, artifacts, run_id, "gaia2_tools", task_server(artifacts, "gaia"), timeout=240)
    result.pop("text")
    calls = [json.loads(line) for line in (artifacts / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    writes = [c for c in calls if c.get("tool") == "Calendar__add_calendar_event" and c.get("arguments", {}).get("title") == title]
    reads = [c for c in calls if c.get("tool") == "Calendar__search_events"]
    aui = [c for c in calls if c.get("tool") == "AgentUserInterface__send_message_to_user"]
    passed = len(writes) == 1 and len(reads) >= 2 and title not in json.dumps(reads[0].get("result")) and title in json.dumps(reads[-1].get("result")) and bool(aui)
    return {**result, "passed": passed, "task_id": "scenario_universe_29_096dyu", "split": "train", "official_score": False,
        "check": "Real model called native ARE tools, read-after-write and AUI; original scenario objective and judge not evaluated", "tool_calls": len([c for c in calls if "tool" in c])}


def swe(artifacts, run_id):
    import docker
    from .swe_adapter import load_row, grade_patch
    manifest = json.loads((PACKAGE / "manifests/swe_bench_pro.json").read_text())
    row = load_row(manifest["train"][0])
    client = docker.from_env(timeout=600)
    image = "jefzda/sweap-images:" + row["dockerhub_tag"]
    client.images.get(image)  # Already warmed; fail before model work if unavailable.
    container = client.containers.run(image, entrypoint="/bin/sh", command=["-c", "sleep infinity"], detach=True,
        name="autoharness-acp-swe-" + uuid.uuid4().hex[:10], mem_limit="8g", pids_limit=512)
    artifacts.mkdir(parents=True, exist_ok=True)
    try:
        location = container.exec_run(["git", "rev-parse", "--show-toplevel"])
        if location.exit_code:
            raise RuntimeError("No SWE task repository")
        repo = location.output.decode().strip()
        reset = container.exec_run(["git", "reset", "--hard", row["base_commit"]], workdir=repo)
        if reset.exit_code:
            raise RuntimeError("Cannot reset SWE base")
        prompt = "Solve this SWE-Bench Pro training task inside the disposable container, using only acp_smoke_task.execute. The tool's working directory is the repository root. Inspect code, edit files, and run focused checks. Do not commit, inspect git history, or read reference patches/hidden tests. You have at most 20 shell calls and five minutes; keep changes focused.\n\n"
        for key in ("problem_statement", "requirements", "interface"):
            prompt += key + ":\n" + str(row.get(key, "")) + "\n\n"
        result = run_acp(prompt, artifacts, run_id, "swe_bench_pro", task_server(artifacts, "docker", container.name, repo), timeout=300)
        result.pop("text")
        container.exec_run(["git", "add", "--intent-to-add", "."], workdir=repo)
        diff = container.exec_run(["git", "diff", "--binary", row["base_commit"]], workdir=repo)
        if diff.exit_code:
            raise RuntimeError("Cannot export generated patch")
        patch = diff.output.decode("utf-8")
        (artifacts / "patch.diff").write_text(patch, encoding="utf-8")
    finally:
        container.remove(force=True)
        client.close()
    grading = grade_patch(row, patch, artifacts / "official_grading")
    return {**result, "passed": True, "task_solved": bool(grading["score"]), "grading": grading, "split": "train", "task_id": row["instance_id"], "official_score": True,
        "check": "Single Codex ACP training-task run with fresh official SWE V1 grading; not H0/H1 or OpenCode"}


def terminal(artifacts, run_id):
    from harbor.models.job.config import JobConfig
    task_id = "log-summary-date-ranges"
    manifest = json.loads((PACKAGE / "manifests/terminal_bench_2.json").read_text())
    if task_id not in manifest["train"]:
        raise ValueError("TB check must remain in train")
    artifacts.mkdir(parents=True, exist_ok=True)
    config = JobConfig.model_validate({"job_name": "trial", "jobs_dir": str(artifacts), "n_attempts": 1, "n_concurrent_trials": 1,
        "skip_credential_checks": True, "environment": {"type": "docker", "delete": True},
        "agents": [{"import_path": "experiments.autosaddler_reproduction.acp_checks:ACPCheckAgent", "model_name": MODEL,
            "skip_model_parsing": True, "kwargs": {"run_id": run_id}}],
        "tasks": [{"path": str(ROOT / "third_party/terminal-bench-2" / task_id)}]})
    config_path = artifacts / "harbor_job.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    completed = subprocess.run([str(ROOT / ".venv-experiment/Scripts/harbor.exe"), "run", "--config", str(config_path)],
        env=env, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1500)
    (artifacts / "harbor_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (artifacts / "harbor_stderr.log").write_text(completed.stderr, encoding="utf-8")
    trials = list((artifacts / "trial").glob("*/result.json"))
    if completed.returncode or len(trials) != 1:
        raise RuntimeError("Harbor check failed; inspect saved logs")
    trial = json.loads(trials[0].read_text(encoding="utf-8"))
    if trial.get("exception_info"):
        raise RuntimeError("Harbor infrastructure error: " + str(trial["exception_info"]))
    rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
    if "reward" not in rewards:
        raise RuntimeError("No native Terminal-Bench reward")
    acp_files = list(artifacts.rglob("acp_result.json"))
    if len(acp_files) != 1:
        raise RuntimeError("Missing ACP usage report")
    native = json.loads(acp_files[0].read_text(encoding="utf-8"))
    return {**native, "passed": True, "task_solved": rewards["reward"] == 1, "score": rewards["reward"], "split": "train", "task_id": task_id,
        "official_score": True, "check": "Single Codex ACP training-task run with Harbor/native verifier; not H0/H1 or OpenCode"}


# Harbor runs this host-side agent while all task commands execute in its container.
from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities


class ACPCheckAgent(BaseAgent):
    capabilities = AgentCapabilities(windows=True)

    def __init__(self, *args, run_id, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_run_id = run_id

    @staticmethod
    def name():
        return "codex-acp-smoke"

    def version(self):
        return "0.1.0"

    async def setup(self, environment):
        return

    async def run(self, instruction, environment, context):
        # Use Harbor's own compose project to identify exactly this task container.
        from harbor.constants import MAIN_SERVICE_NAME
        located = await environment._run_docker_compose_command(["ps", "-q", MAIN_SERVICE_NAME])
        container = located.stdout.strip()
        if not container or "\n" in container:
            raise RuntimeError("Cannot identify unique Harbor task container")
        artifacts = self.logs_dir / "acp"
        prompt = "Solve the following Terminal-Bench training task in the disposable Linux container. Use ONLY acp_smoke_task.execute for shell and file operations. Do not use host tools or inspect /tests, /solution, /logs or evaluation artifacts. At most 20 commands and five minutes. Stop after producing the requested output.\n\n" + instruction
        result = await asyncio.to_thread(run_acp, prompt, artifacts, self.check_run_id, "terminal_bench_2", task_server(artifacts, "docker", container), 300)
        context.n_input_tokens = result["usage"]["input_tokens"]
        context.n_output_tokens = result["usage"]["output_tokens"]
        context.n_cache_tokens = result["usage"]["cached_input_tokens"]
        context.metadata = {"agent": "codex-acp", "infrastructure_only": True, "is_opencode_result": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["protocol", "gaia", "swe", "terminal"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("Run ID must contain only letters, digits, hyphens and underscores")
    artifacts = ROOT / "results/autosaddler_reproduction/acp_checks" / args.run_id / args.action
    if (artifacts / "result.json").exists():
        parser.error("Result already exists; choose a new run ID rather than spending quota again")
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = globals()[args.action](artifacts, args.run_id)
        result["started_at_utc"] = started
    except Exception as exc:
        result = {"passed": False, "error_type": type(exc).__name__, "error": str(exc), "started_at_utc": started, "is_opencode_benchmark_result": False}
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
