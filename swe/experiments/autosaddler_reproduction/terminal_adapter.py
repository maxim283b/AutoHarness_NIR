"""Harbor integration that installs the SAME candidate files used elsewhere."""
from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
import subprocess
import tomllib
import uuid

from harbor.agents.installed.opencode import OpenCode, OpenCodeOptions
from harbor.models.job.config import JobConfig

from .swe_adapter import record_cli_usage

ROOT = Path(__file__).resolve().parents[2]


def git_stdin_compat_script(git_path):
    """Drain check-ignore stdin before Git can reject a non-repository worktree.

    OpenCode 1.19.5 snapshots write paths to this command even with FAKE_VCS.
    Git can exit before reading, triggering an uncaught EPIPE in its caller.
    Preserve Git's exact arguments, output and exit code; do not create a repo.
    """
    return "#!/bin/sh\n" + "git_binary=" + shlex.quote(git_path) + r'''
check_ignore=0
stdin_paths=0
for arg in "$@"; do
    [ "$arg" = check-ignore ] && check_ignore=1
    [ "$arg" = --stdin ] && stdin_paths=1
done
if [ "$check_ignore" = 1 ] && [ "$stdin_paths" = 1 ]; then
    input=$(mktemp) || exit 1
    trap 'rm -f "$input"' EXIT
    cat > "$input" || exit 1
    "$git_binary" "$@" < "$input"
    exit $?
fi
exec "$git_binary" "$@"
'''


def cleanup_command(scope):
    """Stop the live agent command tree; preserve services from completed tools."""
    script = r'''scope=__SCOPE__
root=$(cat /logs/agent/autoharness_agent.pid 2>/dev/null || true)
case "$root" in ''|*[!0-9]*) root=0;; esac
pids=''
if [ "$root" -gt 1 ] && grep -zFxq "AUTOHARNESS_AGENT_SCOPE=$scope" "/proc/$root/environ" 2>/dev/null; then
    kill -STOP "$root" 2>/dev/null || true
    # Stop the current tree before killing its parents. setsid changes session
    # IDs, not ancestry; completed background services reparented to PID 1 stay.
    for round in 1 2 3; do
        pids=$(
            for p in /proc/[0-9]*; do
                [ -r "$p/status" ] || continue
                parent=$(sed -n 's/^PPid:[[:space:]]*//p' "$p/status" 2>/dev/null)
                [ -n "$parent" ] && printf '%s %s\n' "${p##*/}" "$parent"
            done | awk -v root="$root" '{ parent[$1]=$2 } END { live[root]=1; for(i=0;i<NR;i++) for(p in parent) if(live[parent[p]]) live[p]=1; for(p in live) if(live[p] && p>1) print p }'
        )
        for pid in $pids; do kill -STOP "$pid" 2>/dev/null || true; done
    done
    for pid in $pids; do kill -KILL "$pid" 2>/dev/null || true; done
    sleep 0.2
    for pid in $pids; do
        if [ -r "/proc/$pid/status" ] && ! grep -q '^State:.*Z' "/proc/$pid/status"; then
            printf '%s\n' 'Agent cleanup failed' >&2
            exit 1
        fi
    done
fi
mkdir -p /logs/agent
printf '%s\n' '{"cleanup_ok":true,"version":2}' > /logs/agent/autoharness_cleanup.json
'''
    return script.replace("__SCOPE__", shlex.quote(scope))


def trial_outcome(trial, *, install_only=False, cleanup_ok=False):
    detail = trial.get("exception_info") or {}
    timed_out = detail.get("exception_type") == "AgentTimeoutError"
    if detail and (not timed_out or install_only):
        raise RuntimeError("Harbor task infrastructure error: " + str(detail.get("exception_type")) + ": " + str(detail.get("exception_message", ""))[-1500:])
    if install_only:
        return {"installed": True, "valid_evaluation": False}
    if not cleanup_ok:
        raise RuntimeError("Agent process cleanup was not confirmed before verification")
    rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
    if "reward" not in rewards:
        raise RuntimeError("Harbor verifier did not return a reward")
    return {"score":float(rewards["reward"]), "valid_evaluation":True, "agent_timed_out":timed_out, "agent_cleanup_verified":True}


def check_agent_errors(text):
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "error":
            raise RuntimeError("OpenCode reported an agent/API error; this is not a valid benchmark score")


class SharedOpenCodeOptions(OpenCodeOptions):
    candidate_dir: str
    accounting_context: dict[str, str] = {}


class SharedOpenCode(OpenCode):
    options_model = SharedOpenCodeOptions

    def __init__(self, *args, candidate_dir: str, accounting_context=None, **kwargs):
        self.candidate_dir = Path(candidate_dir).resolve()
        super().__init__(*args, candidate_dir=candidate_dir, accounting_context=accounting_context or {}, **kwargs)
        self._opencode_config = json.loads((self.candidate_dir / "opencode.json").read_text(encoding="utf-8"))
        self._opencode_config["instructions"] = ["/opt/autoharness/harness/instructions.md"]
        self._opencode_config.setdefault("provider", {})["opencode-go"] = {"options": {"apiKey": "{env:OPENCODE_GO_API_KEY}"}}
        from .go_gateway import provider_config as go_provider
        self._opencode_config["provider"].update(go_provider(self.model_name or "", accounting_context))
        if self.model_name and self.model_name.startswith("opencode-go/") and os.environ.get("GO_GATEWAY_URL"):
            from .context_policy import configure
            configure(self._opencode_config, self.model_name, "file:///opt/autoharness/context-policy.mjs", "/logs/agent/context-outputs")
        if self.model_name and self.model_name.startswith("codex-acp/"):
            from .acp_gateway import provider_config
            self._opencode_config["provider"].update(provider_config(self.model_name, headers=accounting_context))

    async def exec_as_agent(self, environment, command, env=None, **kwargs):
        marked_env = dict(env or {})
        command = 'export PATH="/opt/autoharness/git-compat:$PATH"; ' + command
        if getattr(self, "_agent_scope", None):
            marked_env["AUTOHARNESS_AGENT_SCOPE"] = self._agent_scope
            command = "mkdir -p /logs/agent; echo $$ > /logs/agent/autoharness_agent.pid; " + command
        return await super().exec_as_agent(environment, command, env=marked_env, **kwargs)

    async def run(self, instruction, environment, context):
        self._agent_scope = uuid.uuid4().hex
        try:
            return await super().run(instruction, environment, context)
        finally:
            async def stop():
                result = await environment.exec(command=cleanup_command(self._agent_scope), user="root", timeout_sec=30)
                if result.return_code:
                    raise RuntimeError("Agent process cleanup failed; verification is forbidden")
            cleanup = asyncio.create_task(stop())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise

    async def install(self, environment):
        platform = await environment.exec(command="ldd --version 2>&1 | head -n 1")
        if "musl" in (platform.stdout or "").lower():
            await super().install(environment)
        else:
            # Same pinned native executable as SWE/GAIA, avoiding npm/nvm setup
            # and repeated downloads in each fresh benchmark container.
            await environment.exec(command="mkdir -p /usr/local/bin")
            await environment.upload_file(ROOT / "third_party/opencode-1.19.5-linux/opencode", "/usr/local/bin/opencode")
            version = await environment.exec(command="chmod +x /usr/local/bin/opencode && opencode --version")
            if version.return_code or "1.19.5" not in (version.stdout or ""):
                raise RuntimeError("Pinned OpenCode binary did not start in the task container")
        await environment.exec(command="mkdir -p /opt/autoharness/harness ~/.config/opencode")
        if self.model_name and self.model_name.startswith("opencode-go/"):
            from .context_policy import PLUGIN
            installed = await environment.exec(command="printf %s " + shlex.quote(PLUGIN) + " > /opt/autoharness/context-policy.mjs")
            if installed.return_code:
                raise RuntimeError("Cannot install context policy plugin")
        git = await environment.exec(command="command -v git")
        if git.return_code == 0 and (git.stdout or "").strip():
            script = git_stdin_compat_script(git.stdout.strip())
            installed = await environment.exec(command="mkdir -p /opt/autoharness/git-compat && printf %s " + shlex.quote(script) + " > /opt/autoharness/git-compat/git && chmod +x /opt/autoharness/git-compat/git")
            if installed.return_code:
                raise RuntimeError("Cannot install Git stdin compatibility wrapper")
        await environment.upload_dir(self.candidate_dir, "/opt/autoharness/harness")
        if (self.candidate_dir / ".opencode").is_dir():
            await environment.exec(command="cp -r /opt/autoharness/harness/.opencode/. ~/.config/opencode/")


class PrepareOnlyOpenCode(SharedOpenCode):
    async def run(self, *args, **kwargs):
        raise RuntimeError("Inference is forbidden in environment-only preflight")


def run_task(task_id, candidate, model, artifacts, ledger, run_id, role, install_only=False):
    from .acp_gateway import context_headers
    gateway_model = model.startswith("codex-acp/")
    config = JobConfig.model_validate({
        "job_name": "trial", "jobs_dir": str(artifacts), "n_attempts": 1,
        "n_concurrent_trials": 1, "install_only": install_only, "skip_credential_checks": install_only or gateway_model,
        "environment": {"type": "docker", "delete": True},
        "agents": [{"import_path": "experiments.autosaddler_reproduction.terminal_adapter:" + ("PrepareOnlyOpenCode" if install_only else "SharedOpenCode"),
            "model_name": model, "skip_model_parsing": True,
            "env": {} if install_only else ({"ACP_GATEWAY_TOKEN": "${ACP_GATEWAY_TOKEN}"} if gateway_model else {"OPENCODE_GO_API_KEY": "${OPENCODE_GO_API_KEY}"}),
            "kwargs": {"candidate_dir": str(candidate.resolve()), "accounting_context": context_headers(run_id, role, "terminal_bench_2", task_id), "version": "1.19.5", "disable_web_search": True}}],
        "tasks": [{"path": str(ROOT / "third_party/terminal-bench-2" / task_id)}],
    })
    artifacts.mkdir(parents=True, exist_ok=True)
    config_path = artifacts / "harbor_job.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    executable = ROOT / ".venv-experiment/Scripts/harbor.exe"
    env = os.environ.copy()
    if install_only:
        for key in ("OPENCODE_GO_API_KEY", "OPENCODE_API_KEY", "ACP_GATEWAY_TOKEN", "GO_GATEWAY_URL"):
            env.pop(key, None)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    task_config = tomllib.loads((ROOT / "third_party/terminal-bench-2" / task_id / "task.toml").read_text(encoding="utf-8"))
    # Some official tasks allow a two-hour agent run; the outer process must not
    # kill them after one hour and misreport an infrastructure failure.
    outer_timeout = sum(float(task_config.get(section, {}).get(field, 900)) for section, field in
        (("agent", "timeout_sec"), ("verifier", "timeout_sec"), ("environment", "build_timeout_sec"))) + 600
    for preflight_attempt in range(3):
        result = subprocess.run([str(executable), "run", "--config", str(config_path)], cwd=ROOT, env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=outer_timeout)
        # Harbor's ten-second `docker info` check can time out on this Windows
        # host. Only retry its exact pre-agent exit, never an actual trial.
        before_any_trial = not (artifacts / "trial").exists()
        daemon_exit = (result.stdout + result.stderr).strip() == "Docker daemon is not running. Please start Docker and try again."
        if not (result.returncode and before_any_trial and daemon_exit and preflight_attempt < 2):
            break
        (artifacts / f"docker_preflight_retry_{preflight_attempt+1}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    (artifacts / "harbor_stdout.log").write_text(result.stdout, encoding="utf-8")
    (artifacts / "harbor_stderr.log").write_text(result.stderr, encoding="utf-8")
    agent_traces = []
    for trace in artifacts.rglob("opencode.txt"):
        text = trace.read_text(encoding="utf-8")
        record_cli_usage(text, ledger, run_id, role, benchmark="terminal_bench_2", task_id=task_id)
        agent_traces.append(text)
    for text in agent_traces:
        check_agent_errors(text)
    trials = [json.loads(p.read_text(encoding="utf-8")) for p in (artifacts / "trial").glob("*/result.json")]
    if result.returncode or len(trials) != 1:
        raise RuntimeError("Harbor did not produce exactly one trial result")
    trial = trials[0]
    cleanup_files = list((artifacts / "trial").glob("*/agent/autoharness_cleanup.json"))
    cleanup_ok = len(cleanup_files) == 1 and json.loads(cleanup_files[0].read_text()).get("cleanup_ok") is True
    outcome = trial_outcome(trial, install_only=install_only, cleanup_ok=cleanup_ok)
    if not install_only:
        outcome["trial_result"] = str(next((artifacts / "trial").glob("*/result.json")))
    return outcome
