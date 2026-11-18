"""Local OpenAI-compatible transport backed by signed-in Codex over ACP.

OpenCode executes the task tools. Codex selects actions through typed MCP stubs
which only record intent, with native command execution disabled. This transport is an experimental
model adapter and is recorded as such, never presented as native model inference.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import re
import shutil
import threading
import time
import uuid

from .accounting import UsageLedger
from .acp_client import ACPClient
from .acp_usage import normalize_acp

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).parent
MODEL = "gpt-6-luna"


def parse_decision(text, tools):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) - {"content", "tool_calls"}:
        raise ValueError("Backend must return exactly an assistant decision object")
    content = value.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("Assistant content must be text or null")
    calls = value.get("tool_calls", [])
    if not isinstance(calls, list) or (not content and not calls):
        raise ValueError("Empty or invalid assistant decision")
    available = {tool["function"]["name"]: tool["function"] for tool in tools or []}
    normalized = []
    from jsonschema import Draft202012Validator
    for index, call in enumerate(calls):
        if not isinstance(call, dict) or call.get("name") not in available:
            raise ValueError("Backend selected an unavailable tool")
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
        Draft202012Validator(available[call["name"]].get("parameters", {})).validate(arguments)
        normalized.append({"index": index, "id": "call_" + uuid.uuid4().hex[:20], "type": "function",
            "function": {"name": call["name"], "arguments": json.dumps(arguments, ensure_ascii=False)}})
    return {"content": content, "tool_calls": normalized}


def provider_config(model, base_url=None, headers=None):
    if not model.startswith("codex-acp/"):
        return {}
    endpoint = base_url or os.environ.get("ACP_GATEWAY_URL")
    if not endpoint or not os.environ.get("ACP_GATEWAY_TOKEN"):
        raise RuntimeError("Start the authorized local ACP gateway before model work")
    if model != "codex-acp/" + MODEL:
        raise ValueError("This experiment freezes the gateway model to " + MODEL)
    return {"codex-acp": {"npm": "@ai-sdk/openai-compatible", "name": "Codex through local ACP model adapter",
        "options": {"baseURL": endpoint, "apiKey": "{env:ACP_GATEWAY_TOKEN}", "timeout": 900000, "maxRetries": 0, "headers": headers or {}},
        "models": {MODEL: {"name": MODEL + " via ACP", "tool_call": True, "attachment": True, "modalities": {"input": ["text", "image"], "output": ["text"]}, "limit": {"context": 200000, "output": 16384}}}}}


def conversation_blocks(payload):
    images = []
    def convert(value):
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            if value.get("type") == "image_url":
                url = value["image_url"]
                url = url.get("url") if isinstance(url, dict) else url
                if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                    raise ValueError("ACP transport requires inline image data, not a remote image URL")
                prefix, data = url.split(",", 1)
                mime = prefix[5:].split(";", 1)[0]
                images.append({"type": "image", "mimeType": mime, "data": data})
                return {"type": "text", "text": "[Image attachment " + str(len(images)) + " supplied with this ACP prompt]"}
            return {key: convert(item) for key, item in value.items()}
        return value
    conversation = {"messages": convert(payload["messages"]), "tool_choice": payload.get("tool_choice", "auto")}
    prompt = "Continue this complete OpenCode conversation. Select at most ONE action through an opencode_actions MCP tool and then end your turn, or give the final assistant answer if the task is finished. Selection only records intent; OpenCode will execute it.\n" + json.dumps(conversation, ensure_ascii=False)
    return [{"type": "text", "text": prompt}, *images]


def context_headers(run_id, role, benchmark=None, task_id=None):
    values = {"X-Autoharness-Run": run_id, "X-Autoharness-Role": role, "X-Autoharness-Benchmark": benchmark, "X-Autoharness-Task": task_id}
    return {key: str(value) for key, value in values.items() if value is not None}


def allow_action_selection(params, events):
    if not params.get("_meta", {}).get("is_mcp_tool_approval"):
        return None
    call_id = params.get("toolCall", {}).get("toolCallId")
    if not any(event.get("params", {}).get("update", {}).get("toolCallId") == call_id and event.get("params", {}).get("update", {}).get("rawInput", {}).get("server") == "opencode_actions" for event in events):
        return None
    return next((option["optionId"] for option in params.get("options", []) if option.get("kind") == "allow_once"), None)


class ACPGateway:
    def __init__(self, artifacts, max_requests=1000, max_tokens=100000000, timeout=180, minimum_weekly_remaining=None):
        self.artifacts = Path(artifacts).resolve()
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.workspace = ROOT / ".test_tmp/acp-model-gateway" / uuid.uuid4().hex[:12]
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.max_requests, self.max_tokens, self.timeout = max_requests, max_tokens, timeout
        if minimum_weekly_remaining is not None and not 0 <= minimum_weekly_remaining < 100:
            raise ValueError("Weekly reserve must be between 0 and 99 percent")
        self.minimum_weekly_remaining = minimum_weekly_remaining
        self.requests, self.total_tokens = 0, 0
        self.lock = threading.Lock()
        self.halted = None
        self.token = secrets.token_urlsafe(36)
        self.ledger = UsageLedger(self.artifacts / "transport_usage.sqlite")

    def __enter__(self):
        try:
            return self._start()
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _start(self):
        node, codex = shutil.which("node"), shutil.which("codex")
        if not node or not codex:
            raise RuntimeError("Authenticated Codex CLI and Node are required")
        global_config = Path(os.environ.get("CODEX_HOME", str(Path.home()/".codex"))) / "config.toml"
        # A clean local profile prevents desktop plugins from starting helper
        # processes for every short-lived model session. Reuse only the existing
        # signed-in credential, never account-wide configuration or MCP servers.
        source_home = global_config.parent
        self.private_home = self.workspace / ".codex-backend"
        self.private_home.mkdir()
        auth_source = source_home / "auth.json"
        if not auth_source.is_file():
            raise RuntimeError("Signed-in Codex file authentication is required for the isolated backend")
        shutil.copy2(auth_source, self.private_home / "auth.json")
        configured_mcp = {}
        config = {"model": MODEL, "model_provider": "openai", "forced_login_method": "chatgpt", "model_reasoning_effort": "low", "approval_policy": "never", "sandbox_mode": "read-only", "web_search": "disabled",
            "model_instructions_file": str(PACKAGE / "acp_gateway_native_instructions.md"),
            "developer_instructions": "Select one next action for OpenCode through an opencode_actions MCP tool, then end your turn. These MCP tools only record intent. OpenCode supplies real results in the next conversation. If finished, answer normally. Do not use any other native tools.",
            "mcp_servers": {name: {"enabled": False} for name in configured_mcp},
            "features": {"shell_tool": False, "plugins": False, "remote_plugin": False, "apps": False, "multi_agent": False, "shell_snapshot": False}}
        allowed = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "HOMEDRIVE", "HOMEPATH", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE"}
        env = {key:value for key,value in os.environ.items() if key.upper() in allowed}
        env.update({"CODEX_HOME": str(self.private_home), "CODEX_PATH": codex, "CODEX_CONFIG": json.dumps(config), "INITIAL_AGENT_MODE": "read-only", "AUTOHARNESS_ACP_ISOLATE": "1"})
        for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "DEFAULT_AUTH_REQUEST", "MODEL_PROVIDER"):
            env.pop(name, None)
        adapter = ROOT / "third_party/codex-acp/node_modules/@agentclientprotocol/codex-acp/dist/index.js"
        self.command, self.backend_env = [node, str(adapter)], env
        self.client = ACPClient(self.command, self.workspace, self.artifacts / "acp_transport", env, permission_handler=allow_action_selection)
        self.capabilities = self.client.initialize()
        quota_session = self.client.new_session(cwd=str(self.workspace))
        self.quota_session_id = quota_session["sessionId"]
        self.read_quota("quota_before.json")
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send_json(self, status, value):
                body = json.dumps(value, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    return self.send_json(200, {"ready": owner.halted is None, "requests": owner.requests, "total_tokens": owner.total_tokens})
                if self.headers.get("Authorization") != "Bearer " + owner.token:
                    return self.send_json(401, {"error": "unauthorized"})
                return self.send_json(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "signed-in-codex-acp"}]})

            def do_POST(self):
                if self.headers.get("Authorization") != "Bearer " + owner.token:
                    return self.send_json(401, {"error": "unauthorized"})
                if self.path != "/v1/chat/completions":
                    return self.send_json(404, {"error": "unsupported endpoint"})
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 32000000:
                    return self.send_json(400, {"error": "invalid request size"})
                try:
                    payload = json.loads(self.rfile.read(length))
                    completion = owner.complete(payload, dict(self.headers))
                except Exception as exc:
                    return self.send_json(503, {"error": {"message": str(exc), "type": "acp_gateway_error", "code": "acp_gateway_stopped"}})
                if not payload.get("stream"):
                    return self.send_json(200, completion)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                message = completion["choices"][0]["message"]
                delta = {key: value for key, value in message.items() if value is not None and (key != "tool_calls" or value)}
                base = {key: completion[key] for key in ("id", "model", "created")}
                events = [{**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                    {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": completion["choices"][0]["finish_reason"]}]},
                    {**base, "object": "chat.completion.chunk", "choices": [], "usage": completion["usage"]}]
                for event in events:
                    self.wfile.write(("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
        self.server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.docker_url = self.base_url.replace("127.0.0.1", "host.docker.internal")
        os.environ["ACP_GATEWAY_URL"] = self.docker_url
        os.environ["ACP_GATEWAY_LOCAL_URL"] = self.base_url
        os.environ["ACP_GATEWAY_TOKEN"] = self.token
        return self

    def read_quota(self, filename="quota_latest.json"):
        quota = self.client.prompt(self.quota_session_id, "/status", timeout=30)
        lines = [line for line in quota["text"].splitlines() if any(word in line.lower() for word in ("limit", "left", "reset", "credits", "remaining"))]
        match = re.search(r"Weekly limit:\*\*\s*(\d+)% left", quota["text"], re.IGNORECASE)
        remaining = int(match.group(1)) if match else None
        report = {"lines": lines, "weekly_remaining_percent": remaining, "checked_at_unix": time.time(), "source": "Codex ACP /status; local command without model inference"}
        (self.artifacts / filename).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return remaining

    def complete(self, payload, headers):
        if payload.get("model") not in {MODEL, "codex-acp/" + MODEL}:
            raise ValueError("Unfrozen gateway model requested")
        lower = {key.lower(): value for key, value in headers.items()}
        role = lower.get("x-autoharness-role", "target_search")
        if role not in {"optimizer", "target_search", "target_final_baseline", "target_final_optimized", "judge"}:
            raise ValueError("Unknown accounting role")
        run_id = lower.get("x-autoharness-run", "gateway-wire-smoke")
        with self.lock:
            if self.halted or self.requests >= self.max_requests or self.total_tokens >= self.max_tokens:
                raise RuntimeError(self.halted or "ACP gateway budget exhausted")
            if self.minimum_weekly_remaining is not None:
                remaining = self.read_quota()
                if remaining is None or remaining <= self.minimum_weekly_remaining:
                    self.halted = "Weekly quota reserve reached or quota unavailable"
                    raise RuntimeError(self.halted)
            self.requests += 1
            request_id = "acp-http-" + uuid.uuid4().hex
            artifacts = self.artifacts / "requests" / request_id
            artifacts.mkdir(parents=True)
            (artifacts / "request.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            usage = normalize_acp({})
            session_id = None
            model_client = None
            try:
                # Isolate each completion's native MCP lifetime. Keeping hundreds
                # of persisted Codex threads alive also keeps their subprocesses.
                model_client = ACPClient(self.command, self.workspace, artifacts / "acp", self.backend_env, permission_handler=allow_action_selection)
                model_client.initialize()
                tools = payload.get("tools") or []
                tools_path, actions_path = artifacts / "tools.json", artifacts / "selected_actions.jsonl"
                tools_path.write_text(json.dumps(tools, ensure_ascii=False), encoding="utf-8")
                servers = [{"name": "opencode_actions", "command": str(ROOT / "third_party/Meta-ARE/.venv/Scripts/python.exe"),
                    "args": ["-m", "experiments.autosaddler_reproduction.acp_gateway_actions", "--tools", str(tools_path), "--actions", str(actions_path)],
                    "env": [{"name": "PYTHONPATH", "value": str(ROOT)}, {"name": "PYTHONUTF8", "value": "1"}]}] if tools else []
                session = model_client.new_session(servers, cwd=str(self.workspace))
                session_id = session["sessionId"]
                if tools:
                    inventory = model_client.prompt(session_id, "/mcp", timeout=45)
                    (artifacts / "mcp_inventory.txt").write_text(inventory["text"], encoding="utf-8")
                    if "opencode_actions" not in inventory["text"]:
                        raise RuntimeError("Expected MCP action-selection server is not available")
                # New session inherits the fixed model/effort; no hidden native tools.
                answer = model_client.prompt(session_id, conversation_blocks(payload), timeout=self.timeout)
                (artifacts / "acp_result.json").write_text(json.dumps(answer["result"], indent=2), encoding="utf-8")
                (artifacts / "backend_answer.txt").write_text(answer["text"], encoding="utf-8")
                usage = normalize_acp(answer["result"])
                if not usage["usage_complete"]:
                    raise RuntimeError("Missing ACP backend usage")
                self.total_tokens += usage["total_tokens"]
                native_calls = [e for e in model_client.events if e.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call"]
                if any(event.get("params", {}).get("update", {}).get("rawInput", {}).get("server") != "opencode_actions" for event in native_calls):
                    raise RuntimeError("Backend attempted a non-selection native tool; OpenCode must execute all task actions")
                calls = [json.loads(line) for line in actions_path.read_text(encoding="utf-8").splitlines()] if actions_path.exists() else []
                if tools and not calls and re.search(r"(?:no|not|unavailable|cannot|can.t).{0,80}(?:MCP tool|opencode_actions)|(?:MCP tool|opencode_actions).{0,80}(?:unavailable|not available)", answer["text"], re.IGNORECASE):
                    raise RuntimeError("Backend reported unavailable action tools; invalid transport response, not a task failure")
                decision = parse_decision(json.dumps({"content": None if calls else answer["text"], "tool_calls": calls}), tools)
                message = {"role": "assistant", "content": decision["content"]}
                if decision["tool_calls"]:
                    message["tool_calls"] = decision["tool_calls"]
                result = {"id": request_id, "object": "chat.completion", "created": int(time.time()), "model": MODEL,
                    "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if decision["tool_calls"] else "stop"}],
                    "usage": {"prompt_tokens": usage["input_tokens"], "completion_tokens": usage["output_tokens"], "total_tokens": usage["total_tokens"],
                        "prompt_tokens_details": {"cached_tokens": usage["cached_input_tokens"]}, "completion_tokens_details": {"reasoning_tokens": usage["reasoning_tokens"]}}}
                (artifacts / "response.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                return result
            except Exception as exc:
                self.halted = type(exc).__name__ + ": " + str(exc)
                (artifacts / "error.json").write_text(json.dumps({"error": self.halted}), encoding="utf-8")
                raise
            finally:
                if model_client is not None:
                    model_client.close()
                self.ledger.record(request_id, run_id, role, usage, benchmark=lower.get("x-autoharness-benchmark"), task_id=lower.get("x-autoharness-task"),
                    model=MODEL, provider="codex-acp", accounting_unit="one OpenCode completion backed by one ACP turn")
                (self.artifacts / "status.json").write_text(json.dumps({"requests": self.requests, "total_tokens": self.total_tokens, "halted": self.halted}, indent=2), encoding="utf-8")

    def __exit__(self, *args):
        try:
            if hasattr(self, "server"):
                self.server.shutdown()
                self.server.server_close()
                self.thread.join(timeout=5)
            if hasattr(self, "client"):
                self.client.close()
        finally:
            self.ledger.close()
            if hasattr(self, "private_home"):
                (self.private_home / "auth.json").unlink(missing_ok=True)
            if os.environ.get("ACP_GATEWAY_TOKEN") == self.token:
                for name in ("ACP_GATEWAY_URL", "ACP_GATEWAY_LOCAL_URL", "ACP_GATEWAY_TOKEN"):
                    os.environ.pop(name, None)


def smoke():
    from .opencode_runtime import OpenCodeSession
    artifacts = ROOT / "results/autosaddler_reproduction/acp_opencode_wire" / ("run-" + uuid.uuid4().hex[:8])
    workspace = ROOT / ".test_tmp/acp-opencode-wire" / artifacts.name
    workspace.mkdir(parents=True)
    for name in ("instructions.md", "opencode.json"):
        shutil.copy2(PACKAGE / "harness_seed" / name, workspace / name)
    (workspace / "fixture.txt").write_text("left=17\nright=25\n", encoding="utf-8")
    run_id = artifacts.name
    ledger = UsageLedger(artifacts / "opencode_usage.sqlite")
    with ACPGateway(artifacts / "gateway", max_requests=8, max_tokens=1000000) as gateway:
        providers = provider_config("codex-acp/" + MODEL, gateway.docker_url, context_headers(run_id, "target_search", "fixture", "read-write"))
        with OpenCodeSession(workspace, artifacts / "opencode", "codex-acp/" + MODEL, overlay={"provider": providers}, container=True) as session:
            result = session.prompt("Read /workspace/fixture.txt with a tool. Add left and right, write exactly 42 followed by a newline to /workspace/answer.txt. Verify the written file with a tool, then reply with exactly {\"answer\":42}. Do not use commands or paths outside /workspace.", timeout=600)
            session.export_usage(ledger, run_id, "target_search", benchmark="fixture", task_id="read-write")
        assert (workspace / "answer.txt").read_text().strip() == "42", result["text"]
        native = ledger.totals(run_id)["target_search"]
        transport = gateway.ledger.totals(run_id)["target_search"]
        assert native["total_tokens"] == transport["total_tokens"], "OpenCode and ACP transport usage differ"
        report = {"passed": True, "agent": "OpenCode 1.19.5", "backend": MODEL + " via Codex ACP typed action-selection adapter", "opencode_owns_tools": True,
            "fixture_read_write_verified": True, "model_completions": gateway.requests, "native_usage_matches_transport": True, "usage": native}
        (artifacts / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"artifacts": str(artifacts), **report}, indent=2))
    ledger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.parse_args()
    smoke()
