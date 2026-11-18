"""OpenCode ACP provider in a task-only Docker workspace for AutoSaddler."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import uuid

from autosaddler.v2.providers.base import BaseAgentProvider, TransportOutcome, observe_usage
from .evidence_renderer import EvidenceRenderer as WorkspaceRenderer
from autosaddler.v2.prompting.models import Usage

from .acp_client import ACPClient
from .accounting import normalize_opencode
from .gaia_native_assets import NATIVE_IMAGE_ID, prepare_sdk_mounts
from .gaia_owned_resources import register_container, container_labels, register_native_process, cleanup_container

IMAGE = NATIVE_IMAGE_ID


def export_session(container, session_id, artifacts):
    """Use a regular file: native OpenCode can exit before flushing a pipe."""
    remote = "/tmp/autoharness-export-" + uuid.uuid4().hex + ".json"
    exported = subprocess.run(["docker", "exec", container, "sh", "-c",
        "opencode export " + shlex.quote(session_id) + " > " + shlex.quote(remote)],
        capture_output=True, timeout=45)
    if exported.returncode:
        raise RuntimeError("Native optimizer session export failed")
    path = Path(artifacts) / "native_export.json"
    copied = subprocess.run(["docker", "cp", container + ":" + remote, str(path)],
        capture_output=True, timeout=45)
    if copied.returncode:
        raise RuntimeError("Cannot copy native optimizer session export")
    # Preserve the exact raw artifact even when parsing/validation fails.
    value = json.loads(path.read_bytes())
    if value.get("info", {}).get("id") != session_id or not isinstance(value.get("messages"), list):
        raise RuntimeError("Native optimizer export does not match its session")
    return value


class Transport:
    def __init__(self, model, run_id=None, output_repair_attempts=0):
        if model not in {"opencode-go/kimi-k2.7-code", "opencode-go/glm-5.3"}:
            raise ValueError("Optimizer requires a registered Go model")
        self.model = model
        self.run_id = run_id
        self.output_repair_attempts = output_repair_attempts

    async def run(self, session, timeout_seconds):
        import threading
        from .gaia_optimizer_supervisor import run_session
        cancellation = threading.Event()
        worker = asyncio.create_task(asyncio.to_thread(run_session, self, session, timeout_seconds, cancellation))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # The parent owns a real child process and its journal. Cancellation
            # requests process-tree/resource cleanup before another stage starts.
            cancellation.set()
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    cancellation.set()
                except Exception:
                    break
            try:
                worker.result()
            except Exception:
                pass
            raise

    def _run(self, session, timeout):
        started = time.monotonic()
        artifacts = session.trace_dir or session.workspace / ".autosaddler/acp-trace"
        artifacts.mkdir(parents=True, exist_ok=True)
        config = {"model": self.model, "instructions": ["/workspace/AGENTS.md"], "share": "disabled", "autoupdate": False,
                  "compaction": {"auto": False, "prune": False},
                  "agent": {"title": {"disable": True}, "summary": {"disable": True}}, "permission": {"task": "deny"},
                  "provider": {"opencode-go": {"options": {"apiKey": "{env:OPENCODE_GO_API_KEY}"}}}}
        from .go_gateway import provider_config as go_provider
        from .acp_gateway import context_headers
        config["provider"].update(go_provider(self.model, context_headers(self.run_id, "optimizer")))
        if self.model.startswith("opencode-go/"):
            from .context_policy import configure, write_plugin
            write_plugin(session.workspace / ".autosaddler/context-policy.mjs")
            configure(config, self.model, "file:///workspace/.autosaddler/context-policy.mjs", "/autoharness-context-output")
            # This mounted directory contains this session's preserved tool output.
            # Native read/glob/grep must be able to reopen the paths in excerpts.
            # Keep all other external-directory requests on the native ask policy.
            config["permission"]["external_directory"] = {
                "*": "ask", "/autoharness-context-output": "allow",
                "/autoharness-context-output/*": "allow",
                # Ephemeral container-local scratch; no host /tmp is mounted.
                # Native bash may write directly under /tmp, not just opencode.
                "/tmp": "allow", "/tmp/*": "allow",
            }
            config["enabled_providers"] = ["opencode-go"]
        if self.model.startswith("codex-acp/"):
            from .acp_gateway import provider_config, context_headers
            config["provider"].update(provider_config(self.model, headers=context_headers(self.run_id, "optimizer")))
        env = os.environ.copy()
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
        name = "autoharness-optimizer-" + uuid.uuid4().hex[:12]
        record = register_container(name, artifacts, kind="native-optimizer")
        sdk_workspace = artifacts / "sdk-metadata-only"
        sdk_workspace.mkdir(exist_ok=True)
        sdk_mounts = prepare_sdk_mounts(sdk_workspace, artifacts)[:4]
        command = ["docker", "run", "--pull", "never", *container_labels(record), "--rm", "-i", "--name", name,
            '--network',os.environ['AUTOHARNESS_DOCKER_NETWORK'],'--cap-drop','NET_RAW',
            "--mount", f"type=bind,src={session.workspace.resolve()},dst=/workspace",
            *sdk_mounts, "--mount", f"type=bind,src={Path(__file__).with_name("source_build_linux.py")},dst=/autoharness-build-check.py,readonly", "-e", "OPENCODE_DISABLE_MODELS_FETCH=1",
            "-e", "OPENCODE_GO_API_KEY", "-e", "ACP_GATEWAY_TOKEN", "-e", "OPENCODE_CONFIG_CONTENT", IMAGE,
            "opencode", "acp", "--pure"]
        if self.model.startswith("opencode-go/"):
            # --pure disables explicit plugins too. Keep project configuration
            # isolated, but load the single frozen hook from CONFIG_CONTENT.
            command.remove("--pure")
            output_dir = artifacts / "context-outputs"
            output_dir.mkdir(exist_ok=True)
            command[command.index(IMAGE):command.index(IMAGE)] = ["-e", "OPENCODE_DISABLE_PROJECT_CONFIG=1", "--mount", f"type=bind,src={output_dir.resolve()},dst=/autoharness-context-output"]
        before_candidate = candidate_entries(session.workspace)
        export = None
        try:
            with ACPClient(command, session.workspace, artifacts, env) as client:
                register_native_process(client.process, record)
                client.initialize()
                response = client.new_session(cwd="/workspace")
                session_id = response["sessionId"]
                client.request("session/set_model", {"sessionId": session_id, "modelId": self.model})
                try:
                    answer = client.prompt(session_id, session.task_prompt, timeout=max(1,timeout-(time.monotonic()-started)))
                    for attempt in range(self.output_repair_attempts + 1):
                        if not self.output_repair_attempts:
                            break
                        failure = structured_output_error(session, before_candidate)
                        if failure is None:
                            break
                        (artifacts / ('structured_output_check_'+str(attempt)+'.json')).write_text(json.dumps({'error':failure}),encoding='utf-8')
                        if attempt == self.output_repair_attempts:
                            raise RuntimeError('Required optimizer output remains invalid: '+failure)
                        remaining = timeout-(time.monotonic()-started)
                        if remaining <= 1:
                            raise TimeoutError('Optimizer deadline reached before structured-output repair')
                        repaired = client.prompt(session_id,
                            'The current stage has no valid final JSON artifact. Complete this SAME stage output and candidate-validity contract, including fixing any reported candidate compilation/schema/path error; do not start another search iteration. Preserve immutable evaluation/model/accounting boundaries. Read .autosaddler/session_output_schema.json and write the actual stage result to .autosaddler/session_output.json. Validation error: '+failure,
                            timeout=remaining)
                        answer['text'] += '\n'+repaired['text']
                except BaseException:
                    # Export while the container still exists, before ACPClient
                    # closes its docker process. Preserve the original failure.
                    try:
                        export_session(name, session_id, artifacts)
                    except Exception as exc:
                        (artifacts / "native_export_error.json").write_text(
                            json.dumps({"error_type":type(exc).__name__, "message":str(exc)}), encoding="utf-8")
                    raise
                # Native export supplies per-request model IDs and token counters.
                export = export_session(name, session_id, artifacts)
        finally:
            cleanup_container(record)
        usage = []
        for message in (export or {}).get("messages", []):
            info = message.get("info", {})
            if info.get("role") != "assistant":
                continue
            row = normalize_opencode(info.get("tokens"))
            usage.append(Usage(input_tokens=row["input_tokens"] or 0, output_tokens=row["output_tokens"] or 0,
                cached_input_tokens=row.get("cached_input_tokens", 0), reasoning_tokens=row.get("reasoning_tokens", 0),
                model=info.get("modelID"), role="optimizer", usage_incomplete=not row["usage_complete"],
                provider_metadata={"request_id": info["id"], "native_session_id": session_id, "cache_write_tokens": row.get("cache_write_tokens", 0)}))
        if not usage:
            usage.append(Usage(role="optimizer", usage_incomplete=True))
        if any(item.usage_incomplete for item in usage):
            for item in usage:
                observe_usage(item)
            raise RuntimeError("Incomplete native optimizer usage; reconcile before more model work")
        return TransportOutcome(raw_response=answer["text"], usage=tuple(usage))


def candidate_entries(workspace):
    """Early contract check only. The unmodified core remains authoritative."""
    import hashlib, stat
    workspace = Path(workspace).resolve()
    entries = {}
    for relative in ('instructions.md', 'opencode.json', '.opencode', 'opencode-src'):
        root = workspace / relative
        if root.is_symlink():
            raise ValueError('Candidate symlink is forbidden')
        paths = [root] if root.is_file() else sorted(root.rglob('*')) if root.is_dir() else []
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(workspace):
                raise ValueError('Candidate path escapes workspace')
            if path.is_file():
                entries[path.relative_to(workspace).as_posix()] = (
                    hashlib.sha256(path.read_bytes()).hexdigest(), bool(path.stat().st_mode & stat.S_IXUSR))
    return entries


def structured_output_error(session, before_candidate=None):
    from jsonschema import validate, ValidationError
    try:
        value=json.loads(session.output_path.read_text(encoding='utf-8'))
        schema=json.loads(session.output_schema_path.read_text(encoding='utf-8'))
        validate(value,schema)
        from .evidence_renderer import validate_receipt
        validate_receipt(session, value)
        if before_candidate is not None and 'changed_paths' in value:
            after = candidate_entries(session.workspace)
            observed = sorted(p for p in before_candidate.keys() | after.keys()
                              if before_candidate.get(p) != after.get(p))
            if set(value['changed_paths']) != set(observed):
                return ('changed_paths must exactly match actual workspace-relative candidate delta; '
                        + 'declared=' + repr(value['changed_paths']) + ', observed=' + repr(observed))
            # Compile inside this SAME logical session, before it closes. A
            # compiler diagnostic shares the existing single JSON/path repair;
            # it creates no extra repair, retry, budget or task observation.
            if any(path.startswith('opencode-src/') for path in observed):
                from .source_build import build_candidate, CandidateCompilationError
                try:
                    _, build = build_candidate(session.workspace)
                except CandidateCompilationError as error:
                    return str(error)
                if session.trace_dir:
                    (session.trace_dir / 'candidate_source_build.json').write_text(json.dumps(build, indent=2), encoding='utf-8')
        return None
    except (OSError,ValueError,ValidationError) as exc:
        return str(exc)[:1200]


def create_provider(*, ledger, settings, run_id=None):
    renderer = WorkspaceRenderer(provider="opencode_acp", instruction_file="AGENTS.md", skill_directory=".agents/skills",
        capability_tools={"read_workspace": ("read", "glob", "grep"), "edit_workspace": ("edit", "write"), "run_commands": ("bash",), "load_skills": ("skill",)})
    return BaseAgentProvider(renderer, Transport(settings.get("model"), run_id,settings.get('output_repair_attempts',0)))
