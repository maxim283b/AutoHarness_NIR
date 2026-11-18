"""Shared pinned OpenCode ACP runtime and native-message usage export."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

from .acp_client import ACPClient
from .accounting import UsageLedger, normalize_opencode

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT.parent / "tbfix/third_party/opencode-1.19.5/opencode.exe"


def candidate_hash(root: Path) -> str:
    from .source_build import candidate_files
    return hashlib.sha256(json.dumps(candidate_files(root), sort_keys=True).encode()).hexdigest()


def isolated_environment(state_dir: Path, config: Path) -> dict:
    # Do not propagate unrelated provider secrets into an agent process.
    allowed = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "HOMEDRIVE", "HOMEPATH", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE", "OPENCODE_API_KEY", "OPENCODE_GO_API_KEY", "ACP_GATEWAY_TOKEN"}
    env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
    for kind in ("CACHE", "DATA", "STATE", "CONFIG"):
        env[f"XDG_{kind}_HOME"] = str(state_dir / kind.lower())
    env.update({"OPENCODE_CONFIG": str(config), "OPENCODE_DISABLE_AUTOUPDATE": "true", "PYTHONUTF8": "1"})
    return env


class OpenCodeSession:
    def __init__(self, workspace: Path, artifacts: Path, model: str, mcp_servers=None, overlay=None, container=False):
        if not model or "/" not in model:
            raise ValueError("Explicit provider/model is required")
        self.workspace, self.artifacts, self.model = workspace, artifacts, model
        artifacts.mkdir(parents=True, exist_ok=True)
        config = json.loads((workspace / "opencode.json").read_text(encoding="utf-8"))
        config.update(overlay or {})
        config["model"] = model
        self.config_path = artifacts / "runtime-opencode.json"
        self.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        self.env = isolated_environment(artifacts / "opencode-state", self.config_path)
        self.env["OPENCODE_CONFIG_DIR"] = str(workspace / ".opencode")
        self.container_name = "autoharness-target-" + uuid.uuid4().hex[:12] if container else None
        if container:
            self.env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
            command = ["docker", "run", "--rm", "-i", "--name", self.container_name,
                "--mount", f"type=bind,src={workspace.resolve()},dst=/workspace",
                "-e", "OPENCODE_GO_API_KEY", "-e", "ACP_GATEWAY_TOKEN", "-e", "OPENCODE_CONFIG_CONTENT", "-e", "OPENCODE_CONFIG_DIR=/workspace/.opencode",
                "autoharness-opencode:1.19.5", "opencode", "acp", "--pure"]
        else:
            command = [str(BIN), "acp", "--pure"]
        self.client = ACPClient(command, workspace, artifacts, self.env)
        try:
            self.capabilities = self.client.initialize()
            self.session = self.client.new_session(mcp_servers, cwd="/workspace" if container else None)
            self.session_id = self.session["sessionId"]
            # ACP sessions may otherwise select their own default model.
            self.client.request("session/set_model", {"sessionId": self.session_id, "modelId": model})
        except Exception:
            self.client.close()
            raise

    def prompt(self, text, timeout=900):
        return self.client.prompt(self.session_id, text, timeout)

    def export_usage(self, ledger: UsageLedger, run_id: str, role: str, **context):
        command = ["docker", "exec", self.container_name, "opencode", "export", self.session_id] if self.container_name else [str(BIN), "export", self.session_id]
        result = subprocess.run(command, cwd=self.workspace,
            env=self.env, capture_output=True, text=True, encoding="utf-8", timeout=60)
        if result.returncode:
            ledger.record(f"{self.session_id}:unknown", run_id, role, normalize_opencode(None), **context)
            raise RuntimeError("OpenCode usage export failed; do not continue paid execution without reconciliation")
        exported = json.loads(result.stdout)
        (self.artifacts / "opencode-export.json").write_text(json.dumps(exported, ensure_ascii=False), encoding="utf-8")
        messages = exported.get("messages", [])
        count = 0
        incomplete = False
        for message in messages:
            info = message.get("info", {})
            if info.get("role") != "assistant":
                continue
            usage = normalize_opencode(info.get("tokens"))
            incomplete = incomplete or not usage["usage_complete"]
            usage["provider_cost_usd"] = info.get("cost")
            ledger.record(f"{self.session_id}:{info['id']}", run_id, role, usage,
                provider=info.get("providerID"), model=info.get("modelID"), **context)
            count += 1
        if not count:
            ledger.record(f"{self.session_id}:unknown", run_id, role, normalize_opencode(None), **context)
            raise RuntimeError("No native usage available; stop and reconcile before more model work")
        if incomplete:
            raise RuntimeError("Incomplete native usage; stop and reconcile before more model work")
        return count

    def close(self):
        self.client.close()
        if self.container_name:
            subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True, timeout=30)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
