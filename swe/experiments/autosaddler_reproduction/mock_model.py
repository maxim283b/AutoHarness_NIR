"""Deterministic local wire fixture for real OpenCode; not a real LLM or score."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import uuid


class MockModel:
    def __init__(self, text='{"ready":true}', responder=None):
        self.text, self.responder, self.requests = text, responder, []
    def __enter__(self):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                owner.requests.append(payload)
                message = owner.responder(payload, len(owner.requests)) if owner.responder else {"content": owner.text}
                base = {"id": f"fixture-{len(owner.requests)}", "created": int(time.time()), "model": "fixture"}
                usage = {"prompt_tokens": 37, "completion_tokens": 5, "total_tokens": 42}
                if payload.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    events = [
                        {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", **message}, "finish_reason": None}]},
                        {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}]},
                        {**base, "object": "chat.completion.chunk", "choices": [], "usage": usage},
                    ]
                    for event in events:
                        self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    result = {**base, "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": "stop"}], "usage": usage}
                    body = json.dumps(result).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        return self
    def provider(self):
        return {"mock": {"npm": "@ai-sdk/openai-compatible", "name": "Local protocol fixture", "options": {"baseURL": self.base_url, "apiKey": "fixture-not-a-secret"}, "models": {"fixture": {"name": "Fixture", "limit": {"context": 32768, "output": 1024}}}}}
    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def smoke():
    from pathlib import Path
    import shutil
    from .accounting import UsageLedger
    from .opencode_runtime import OpenCodeSession
    root = Path(__file__).resolve().parents[2]
    workspace = root / ".test_tmp/opencode_mock_wire/workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    for name in ("instructions.md", "opencode.json"):
        shutil.copy2(Path(__file__).with_name("harness_seed") / name, workspace / name)
    artifacts = root / "results/autosaddler_reproduction/opencode_mock_wire"
    ledger = UsageLedger(artifacts / "usage.sqlite")
    run_id = "wire-smoke-" + uuid.uuid4().hex[:8]
    with MockModel() as model:
        with OpenCodeSession(workspace, artifacts, "mock/fixture", overlay={"provider": model.provider()}) as session:
            response = session.prompt("Reply with the fixture JSON.", timeout=90)
            assert json.loads(response["text"]) == {"ready": True}, response["text"]
            session.export_usage(ledger, run_id, "target_search", benchmark="fixture", task_id="fixture")
        totals = ledger.totals(run_id)
        assert totals["target_search"]["calls"] == len(model.requests), "Unaccounted auxiliary model request"
        result = {"passed": True, "run_id": run_id, "real_opencode_version": "1.19.5", "transport": "ACP", "fixture_requests": len(model.requests), "paid_model_requests": 0, "usage": totals}
        (artifacts / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    ledger.close()


if __name__ == "__main__":
    smoke()
