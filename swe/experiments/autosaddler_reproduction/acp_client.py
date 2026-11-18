"""Small ACP v1 stdio client with bounded process lifetime and raw event capture."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time


class ACPError(RuntimeError):
    pass


class ACPClient:
    def __init__(self, command: list[str], cwd: Path, artifact_dir: Path, env: dict | None = None, permission_handler=None):
        self.cwd = cwd.resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = artifact_dir / "acp_events.jsonl"
        self.error_file = (artifact_dir / "acp_stderr.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            command, cwd=self.cwd, env=env or os.environ.copy(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.error_file,
            text=True, encoding="utf-8", bufsize=1,
        )
        self.inbox = queue.Queue()
        self.events: list[dict] = []
        self.counter = 0
        self.pending: dict[int, dict] = {}
        self.write_lock = threading.Lock()
        self.permission_handler = permission_handler
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Native launchers may write startup text.
                self.inbox.put(msg)
        finally:
            self.inbox.put(None)

    def send(self, msg):
        with self.write_lock:
            self.process.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            self.process.stdin.flush()

    def request(self, method: str, params: dict, timeout: float = 120):
        self.counter += 1
        request_id = self.counter
        self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        return self.wait(request_id, timeout)

    def wait(self, request_id, timeout):
        deadline = time.monotonic() + timeout
        while True:
            if request_id in self.pending:
                msg = self.pending.pop(request_id)
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("ACP request timed out")
                try:
                    msg = self.inbox.get(timeout=remaining)
                except queue.Empty:
                    raise TimeoutError("ACP request timed out") from None
                if msg is None:
                    raise ACPError(f"ACP process exited: {self.process.poll()}")
                if not msg.get("method", "").startswith("_auth/"):
                    self.events.append(msg)
                    with self.events_path.open("a", encoding="utf-8") as file:
                        file.write(json.dumps(msg, ensure_ascii=False) + "\n")
            if msg.get("method"):
                self._handle_agent_message(msg)
                continue
            if msg.get("id") != request_id:
                self.pending[msg.get("id")] = msg
                continue
            if "error" in msg:
                raise ACPError(json.dumps(msg["error"], ensure_ascii=False))
            return msg.get("result", {})

    def _handle_agent_message(self, msg):
        if "id" not in msg:
            return
        # No client filesystem or terminal capabilities are advertised. Native agent
        # tools must obey their own configured sandbox; unknown requests fail closed.
        if msg["method"] == "session/request_permission":
            result = {"outcome": {"outcome": "cancelled"}}
            if self.permission_handler:
                option_id = self.permission_handler(msg.get("params", {}), self.events)
                offered = {option.get("optionId") for option in msg.get("params", {}).get("options", []) if option.get("kind") == "allow_once"}
                if option_id in offered:
                    result = {"outcome": {"outcome": "selected", "optionId": option_id}}
            self.send({"jsonrpc": "2.0", "id": msg["id"], "result": result})
        else:
            self.send({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "Client capability not enabled"}})

    def initialize(self):
        return self.request("initialize", {
            "protocolVersion": 1,
            "clientCapabilities": {},
            "clientInfo": {"name": "autoharness-experiment", "version": "0.1.0"},
        })

    def new_session(self, mcp_servers: list[dict] | None = None, cwd: str | None = None):
        return self.request("session/new", {"cwd": cwd or str(self.cwd), "mcpServers": mcp_servers or []})

    def prompt(self, session_id: str, text: str, timeout: float = 180):
        begin = len(self.events)
        blocks = [{"type": "text", "text": text}] if isinstance(text, str) else text
        result = self.request("session/prompt", {"sessionId": session_id, "prompt": blocks}, timeout)
        chunks = []
        for msg in self.events[begin:]:
            update = msg.get("params", {}).get("update", {})
            if update.get("sessionUpdate") == "agent_message_chunk":
                chunks.append(update.get("content", {}).get("text", ""))
        return {"result": result, "text": "".join(chunks), "events": self.events[begin:]}

    def close(self):
        if self.process.poll() is None:
            if os.name == "nt":
                # Windows venv/Node launchers may leave child processes holding pipes.
                subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"], capture_output=True, timeout=15)
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.error_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
