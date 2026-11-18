"""Metered native Go API proxy; credentials stay on the host, never in task containers.

Uses the existing Console OAuth login via the installed Desktop CLI. Native
Anthropic/OpenAI request and response formats are preserved. No model fallback,
retries, token refresh writes or balance top-ups are performed by this proxy.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import select
import socket
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager

import httpx
from .response_evidence import ResponseEvidence

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "qwen3.7-plus": {"family": "anthropic", "sdk": "@ai-sdk/anthropic", "input_bound": 0.50, "output": 1.60},
    "kimi-k2.7-code": {"family": "openai", "sdk": "@ai-sdk/openai-compatible", "input_bound": 0.95, "output": 4.00},
    "glm-5.3": {"family": "openai", "sdk": "@ai-sdk/openai-compatible", "input_bound": 1.40, "output": 4.40},
}

# A native response can exceed its requested output cap (68,883 billed output
# tokens observed). Its next primary/compaction request must retain that text.
# This is a local byte admission bound, not the provider's token context limit.
TEXT_REQUEST_BYTES = 2 * 1024 * 1024
MEDIA_REQUEST_BYTES = 16 * 1024 * 1024
REQUEST_GUARD_VERSION = 3
RESERVATION_VERSION = 2
STREAM_TRANSPORT_VERSION = 1
REQUEST_TIMEOUT_VERSION = 1
# Transport ceiling exceeds the longest official 7200s task deadline. Harbor
# still cancels the native client at each task's unchanged official deadline.
REQUEST_TIMEOUT_SECONDS = 10800
DISCONNECT_WATCH_VERSION = 1
QUOTA_TRANSPORT_ATTEMPTS = 2
# Billing output may exceed the requested generation cap (observed: 36,890
# billed tokens for a request accepted under the 8,192 request guard). This is
# an accounting reserve only; never change the native request to fit it.
BILLED_OUTPUT_RESERVE_TOKENS = 131072


class ClientDisconnected(RuntimeError):
    """An abandoned caller must not launch another inference request."""


@contextmanager
def watch_downstream(response, alive, closing):
    """Cancel an idle upstream read when its native caller has already left."""
    stopped=threading.Event();cancelled=threading.Event()
    def watch():
        while not stopped.wait(.1):
            if not alive() or closing():
                cancelled.set()
                # Closing a response alone may leave a blocking socket read
                # alive. Shutdown the request's own socket, never other calls.
                stream=response.extensions.get('network_stream')
                try:
                    connection=stream.get_extra_info('socket') if stream else None
                    if connection is not None: connection.shutdown(socket.SHUT_RDWR)
                except (OSError,AttributeError):
                    pass
                try: response.close()
                except Exception: pass
                return
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    try:
        yield
    except Exception as error:
        if cancelled.is_set() and not isinstance(error,ClientDisconnected):
            raise ClientDisconnected('Native client left while upstream was idle') from error
        raise
    finally:
        stopped.set();thread.join(timeout=1)


def watched_chunks(response,alive,closing):
    with watch_downstream(response,alive,closing):
        yield from response.iter_bytes()
        if not alive():
            raise ClientDisconnected('Native client disconnected at response end')


def terminal_frame(frame):
    """Hold native stream termination until Go billing has been validated."""
    for line in frame.decode("utf-8").splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return True
        value = json.loads(data)
        if value.get("type") in {"message_stop", "response.completed", "error"}:
            return True
        if any(c.get("finish_reason") is not None for c in value.get("choices", [])):
            return True
    return False


def request_sizes(payload):
    """Separate native binary attachments from text without rewriting the wire.

    The 2 MiB text bound includes schemas, tool arguments and framing.
    Only validated inline image/PDF data in native content blocks is exempt.
    Reservations continue to charge the pessimistic FULL serialized byte bound.
    """
    wire_size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    if wire_size > MEDIA_REQUEST_BYTES:
        raise RuntimeError("Request exceeded 16 MiB native media transport bound")
    text_only = copy.deepcopy(payload)
    allowed = {"image/png", "image/jpeg", "image/webp", "image/gif", "application/pdf"}

    def checked(data, mime):
        if mime not in allowed or not isinstance(data, str):
            raise ValueError("Unsupported inline media attachment")
        try:
            base64.b64decode(data, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid inline media encoding") from exc
        return "[native binary attachment]"

    def content(blocks):
        if not isinstance(blocks, list):
            return
        for block in blocks:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            source = block.get("source", {})
            if kind in {"image", "document"} and isinstance(source, dict) and source.get("type") == "base64":
                source["data"] = checked(source.get("data"), source.get("media_type"))
            elif kind == "image_url":
                image = block.get("image_url", {})
                url = image.get("url", "") if isinstance(image, dict) else ""
                if url.startswith("data:") and ";base64," in url:
                    prefix, data = url.split(";base64,", 1)
                    image["url"] = checked(data, prefix[5:])
            elif kind == "tool_result":
                content(block.get("content"))

    for message in text_only.get("messages", []):
        if isinstance(message, dict):
            content(message.get("content"))
    text_size = len(json.dumps(text_only, ensure_ascii=False).encode("utf-8"))
    if text_size > TEXT_REQUEST_BYTES:
        raise RuntimeError("Request exceeded frozen 2 MiB context safety bound")
    return wire_size, text_size


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def provider_config(model, context=None):
    if not model.startswith("opencode-go/") or not os.environ.get("GO_GATEWAY_URL"):
        return {}
    model_id = model.split("/", 1)[1]
    spec = MODELS[model_id]
    return {"opencode-go": {"npm": spec["sdk"], "name": "Go Plus metered native API",
        "options": {"baseURL": os.environ["GO_GATEWAY_URL"] + "/" + spec["family"] + "/v1",
            "apiKey": "{env:OPENCODE_GO_API_KEY}", "maxRetries": 0, "timeout": REQUEST_TIMEOUT_SECONDS * 1000,
            "headers": context or {}},
        "models": {model_id: {"name": model_id, "tool_call": True, "attachment": True,
            "limit": {"context": 128000, "output": 8192}}}}}


def exported_credential(raw):
    exported = json.loads(raw)
    rows = exported if isinstance(exported, list) else [exported]
    active = [row for row in rows if row.get("integrationID") == "opencode" and row.get("active")]
    if len(active) != 1:
        raise RuntimeError("Expected exactly one active OpenCode Console credential")
    return active[0]["value"]


def credential():
    from .fullsource_host import checked_file
    CLI = checked_file('desktop_cli')
    # The command's output is consumed privately; never log stderr/stdout on error.
    result = subprocess.run([str(CLI), "auth", "export", "OpenCode Console"],
        capture_output=True, encoding="utf-8", timeout=30)
    if result.returncode:
        raise RuntimeError("Cannot read the existing OpenCode Console connection")
    value = exported_credential(result.stdout)
    if value.get("type") != "oauth" or not value.get("metadata", {}).get("orgID"):
        raise RuntimeError("Expected a workspace-scoped OpenCode Console OAuth connection")
    if value.get("expires", 0) / 1000 < time.time() + 60:
        # Resolve through Desktop so it owns rotation of its own credentials.
        refreshed = subprocess.run([str(CLI), "api", "GET", "/api/experimental/usage/limits"],
            capture_output=True, encoding="utf-8", timeout=40)
        if refreshed.returncode:
            raise RuntimeError("Desktop could not refresh its Console connection")
        result = subprocess.run([str(CLI), "auth", "export", "OpenCode Console"],
            capture_output=True, encoding="utf-8", timeout=30)
        if result.returncode:
            raise RuntimeError("Cannot read refreshed Console credential")
        value = exported_credential(result.stdout)
        if value.get("expires", 0) / 1000 < time.time() + 30:
            raise RuntimeError("Console OAuth session is expired")
    return value


def quota(client, auth, ceiling=70):
    response = client.get("https://opencode.ai/zen/go/v1/usage", headers={
        "Authorization": "Bearer " + auth["access"], "x-opencode-org-id": auth["metadata"]["orgID"],
        "User-Agent": "autoharness-opencode/1.19.5"}, timeout=25)
    if response.status_code != 200:
        raise RuntimeError("Go quota endpoint returned HTTP " + str(response.status_code))
    value = response.json()["usage"]
    for window in ("rolling", "weekly", "monthly"):
        row = value[window]
        if row.get("status") != "ok" or isinstance(row.get("percent"), bool) or not isinstance(row.get("percent"), (int, float)) or not math.isfinite(row["percent"]) or not 0 <= row["percent"] < ceiling:
            raise RuntimeError("Go quota guard stopped new work: " + window)
    return value


def reservation(payload, model):
    # Reserve from the full serialized UTF-8 size plus protocol framing; native
    # context admission and tokenization remain the provider's responsibility.
    size, _ = request_sizes(payload)
    output = payload.get("max_tokens", payload.get("max_completion_tokens", 8192))
    if not isinstance(output, int) or not 0 < output <= 8192:
        raise RuntimeError("Request exceeded frozen 8192 output token bound")
    spec = MODELS[model]
    # Allowance can differ from list-price cost (plan normalization). Keep a
    # factor-of-two reserve and fail closed if the real allowance exceeds it.
    return 2 * ((size + 4096) * spec["input_bound"] + max(output, BILLED_OUTPUT_RESERVE_TOKENS) * spec["output"]) / 1000000


def usage_events(raw, streaming):
    if not streaming:
        return [json.loads(raw)]
    result = []
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            result.append(json.loads(line[5:].strip()))
    return result


def metered_usage(events):
    # Do not add cumulative usage at message_start and message_delta twice.
    merged = {}
    for event in events:
        for value in (event.get("usage"), (event.get("message") or {}).get("usage")):
            if isinstance(value, dict):
                merged.update(value)
    billing = merged.get("opencode") or {}
    if billing.get("billing_source") != "go-plus":
        raise RuntimeError("Response did not confirm Go Plus billing; stopping")
    if merged.get("cost") != 0:
        raise RuntimeError("Response reports a balance charge; stopping")
    amount = billing.get("allowance")
    if not isinstance(amount, (float, int)) or not math.isfinite(amount) or amount < 0:
        raise RuntimeError("Response omitted valid Go allowance usage; stopping")
    tokens = billing.get("tokens")
    if not isinstance(tokens, dict) or "input" not in tokens or "output" not in tokens:
        raise RuntimeError("Response omitted native billed token counters; stopping")
    for name in ("input", "output", "reasoning", "cache_read", "cache_write_5m", "cache_write_1h"):
        value = tokens.get(name, 0)
        if not isinstance(value, int) or value < 0:
            raise RuntimeError("Invalid billed token counter")
    return {"allowance": amount, "billing_source": billing["billing_source"], "balance_cost": 0,
        "list_cost": billing.get("list_cost"), "tokens": tokens,
        "total_tokens": sum(tokens.get(k, 0) for k in ("input", "output", "cache_read", "cache_write_5m", "cache_write_1h"))}


class GoGateway:
    def __init__(self, artifacts, allowance_limit=5.0, request_limit=100, role_limits=None, role_request_limits=None, prior_journals=(), require_context_policy=False, optimizer_model="kimi-k2.7-code", quota_ceiling=70):
        if optimizer_model not in {"kimi-k2.7-code", "glm-5.3"}:
            raise ValueError("Unsupported frozen optimizer model")
        if isinstance(quota_ceiling, bool) or not isinstance(quota_ceiling, (int,float)) or not 0 < quota_ceiling <= 100:
            raise ValueError("Invalid quota ceiling")
        self.artifacts = Path(artifacts)
        self.optimizer_model, self.quota_ceiling = optimizer_model, quota_ceiling
        self.allowance_limit, self.request_limit = allowance_limit, request_limit
        self.role_limits = role_limits or {}
        self.role_request_limits = role_request_limits or {}
        self.require_context_policy = require_context_policy
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.client = httpx.Client(follow_redirects=False, timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=30, pool=30))
        self.halted = None
        self.admission_blocked = None
        self.closing = False
        self.active_requests = set()
        self.path = self.artifacts / "requests.jsonl"
        self.rows = []
        history = []
        for journal in prior_journals:
            journal = Path(journal).resolve()
            raw = journal.read_bytes()
            history.append({"path":str(journal), "sha256":hashlib.sha256(raw).hexdigest()})
            self.rows.extend(json.loads(line) for line in raw.decode("utf-8").splitlines())
        history_path = self.artifacts / "budget_history.json"
        if history_path.exists() and json.loads(history_path.read_text()) != history:
            raise RuntimeError("Prior budget history changed; cannot reset previous spending")
        save(history_path, history)
        if self.path.exists():
            self.rows.extend(json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines())
        # Persisted reservations with unknown outcomes remain charged at their bound.
        self.last_quota = None

    def spent(self, role=None):
        final = {r["id"]:r for r in self.rows}
        return sum(r.get("usage", {}).get("allowance", r["reservation"]) for r in final.values() if role is None or r["role"] == role)

    def accounting_summary(self):
        roles = {}
        for row in {r["id"]:r for r in self.rows}.values():
            value = roles.setdefault(row["role"], {"requests":0, "tokens":0, "input_tokens":0, "output_tokens":0, "allowance":0, "unsettled_reserve":0, "balance_cost":0, "unknown_usage_requests":0})
            value["requests"] += 1
            usage = row.get("usage")
            if usage is None:
                value["unknown_usage_requests"] += 1
                value["unsettled_reserve"] += row["reservation"]
            else:
                value["tokens"] += usage["total_tokens"]
                output = usage.get("tokens", {}).get("output", 0)
                value["input_tokens"] += usage["total_tokens"] - output
                value["output_tokens"] += output
                value["allowance"] += usage["allowance"]
                value["balance_cost"] += usage["balance_cost"]
        for value in roles.values():
            if value["unknown_usage_requests"]:
                value["tokens"] = None
                value["input_tokens"] = value["output_tokens"] = None
        # This disjoint breakdown is a view of the same rows, never extra cost.
        groups = {}
        for row in {r["id"]:r for r in self.rows}.values():
            key = (row.get("benchmark") or "shared_search", row["role"], row.get("purpose", "primary"))
            value = groups.setdefault(key, {"benchmark":key[0], "role":key[1], "purpose":key[2],
                "requests":0, "known_tokens":0, "allowance":0, "unsettled_reserve":0, "balance_cost":0, "unknown_usage_requests":0})
            value["requests"] += 1
            usage = row.get("usage")
            if usage is None:
                value["unknown_usage_requests"] += 1
                value["unsettled_reserve"] += row["reservation"]
            else:
                value["known_tokens"] += usage["total_tokens"]
                value["allowance"] += usage["allowance"]
                value["balance_cost"] += usage["balance_cost"]
        return {"includes_prior_runs":True, "roles":roles, "by_benchmark_role_purpose":list(groups.values()),
            "breakdown_is_subset_not_additional_cost":True, "charged_allowance_including_reserves":self.spent()}

    def append(self, row):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        self.rows.append(row)

    def __enter__(self):
        self.last_quota = None
        auth = credential()
        self.last_quota = quota(self.client, auth, self.quota_ceiling)
        save(self.artifacts / "quota_before.json", self.last_quota)
        owner = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def client_alive(self):
                try:
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    return not readable or bool(self.connection.recv(1, socket.MSG_PEEK))
                except OSError:
                    return False

            def start_stream(self, status, headers):
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.flush()
                self.stream_started = True

            def write_chunk(self, chunk):
                if not self.client_alive():
                    raise ClientDisconnected("Native client disconnected")
                try:
                    self.wfile.write(("%x\r\n" % len(chunk)).encode() + chunk + b"\r\n")
                    self.wfile.flush()
                except OSError as exc:
                    raise ClientDisconnected("Native client disconnected") from exc

            def do_POST(self):
                received = datetime.now(timezone.utc).isoformat()
                self.stream_started = False
                given = self.headers.get("x-api-key") or self.headers.get("Authorization", "").removeprefix("Bearer ")
                if not owner.authenticated(given):
                    self.send_error(401)
                    return
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= MEDIA_REQUEST_BYTES:
                    self.send_error(413)
                    return
                try:
                    status, headers, body = owner.forward(self.path, json.loads(self.rfile.read(size)), owner.bound_headers(given, dict(self.headers)),
                        stream_start=self.start_stream, stream_chunk=self.write_chunk,
                        client_alive=self.client_alive, received_at=received)
                except Exception as exc:
                    if self.stream_started or isinstance(exc, ClientDisconnected):
                        self.close_connection = True
                        return
                    status, headers = 400, {"Content-Type": "application/json"}
                    body = json.dumps({"error":{"message":str(exc), "type":"experiment_guard"}}).encode()
                if self.stream_started:
                    try:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                    except OSError:
                        pass
                    return
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:
                    pass
        self.server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        os.environ["GO_GATEWAY_URL"] = self.base_url.replace("127.0.0.1", "host.docker.internal")
        os.environ["OPENCODE_GO_API_KEY"] = self.token
        return self

    def authenticated(self, given):
        return secrets.compare_digest(given, self.token)

    def bound_headers(self, given, headers):
        return headers

    def fresh_quota(self, auth, alive):
        """Retry only metadata transport failures, never inference or quota denials."""
        for attempt in range(1, QUOTA_TRANSPORT_ATTEMPTS + 1):
            if self.closing or not alive():
                raise ClientDisconnected("Caller left before fresh quota validation")
            event = {"timestamp": datetime.now(timezone.utc).isoformat(), "attempt": attempt,
                     "max_attempts": QUOTA_TRANSPORT_ATTEMPTS, "endpoint_kind": "quota_get"}
            try:
                value = quota(self.client, auth, self.quota_ceiling)
            except httpx.TransportError as error:
                event.update(status="transport_error", error_type=type(error).__name__)
                with (self.artifacts / "quota_checks.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(event) + "\n")
                if attempt == QUOTA_TRANSPORT_ATTEMPTS:
                    raise
                time.sleep(1)
            except Exception as error:
                event.update(status="rejected", error_type=type(error).__name__)
                with (self.artifacts / "quota_checks.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(event) + "\n")
                raise
            else:
                event.update(status="validated")
                with (self.artifacts / "quota_checks.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(event) + "\n")
                return value

    def forward(self, path, payload, headers, *, stream_start=None, stream_chunk=None, client_alive=None, received_at=None):
        alive = client_alive or (lambda: True)
        with self.lock:
            if not alive():
                raise ClientDisconnected("Caller disconnected before admission")
            if self.closing:
                raise ClientDisconnected("Gateway is closing")
            if self.halted:
                raise RuntimeError(self.halted)
            if self.admission_blocked:
                raise RuntimeError("Invocation admission suspended; fresh explicit admission required")
            model = payload.get("model")
            if model not in MODELS:
                raise ValueError("Model is outside frozen Go model allowlist")
            family = MODELS[model]["family"]
            expected = "/anthropic/v1/messages" if family == "anthropic" else "/openai/v1/chat/completions"
            if path.split("?", 1)[0] != expected:
                raise ValueError("Unexpected inference endpoint")
            lower = {k.lower():v for k,v in headers.items()}
            role = lower.get("x-autoharness-role")
            if role not in {"optimizer", "target_search", "target_final_baseline", "target_final_optimized"}:
                raise ValueError("Missing or invalid accounting role")
            if (role == "optimizer") != (model == self.optimizer_model):
                raise ValueError("Wrong model for experiment role")
            purpose = lower.get("x-autoharness-purpose", "primary")
            if purpose not in {"primary", "compaction"}:
                raise ValueError("Invalid accounting purpose")
            if self.require_context_policy:
                from .context_policy import VERSION
                expected_policy = VERSION if role == 'optimizer' else 'target-metering-v6'
                if lower.get("x-autoharness-context-policy") != expected_policy or "x-autoharness-purpose" not in lower:
                    raise ValueError("Missing native context policy accounting hook")
            reserved = reservation(payload, model)
            count = len({row["id"] for row in self.rows})
            if count >= self.request_limit or self.spent() + reserved > self.allowance_limit:
                raise RuntimeError("Frozen Go experiment budget reached")
            if role in self.role_limits and self.spent(role) + reserved > self.role_limits[role]:
                raise RuntimeError("Frozen role budget reached: " + role)
            role_count = len({row["id"] for row in self.rows if row["role"] == role})
            if role in self.role_request_limits and role_count >= self.role_request_limits[role]:
                raise RuntimeError("Frozen role request budget reached: " + role)
            # A previous quota success cannot authorize work after a failed refresh.
            # No reservation or POST exists yet at this boundary.
            self.last_quota = None
            try:
                auth = credential()
                self.last_quota = self.fresh_quota(auth, alive)
                save(self.artifacts / "quota_latest.json", self.last_quota)
            except Exception as error:
                self.last_quota = None
                self.admission_blocked = {
                    "blocked_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "quota_wait": isinstance(error, RuntimeError) and str(error).startswith("Go quota guard stopped new work:"),
                    "reservation_created": False,
                    "inference_dispatched": False,
                    "automatic_resume": False,
                }
                save(self.artifacts / "admission_blocked.json", self.admission_blocked)
                raise
            row = {"id":uuid.uuid4().hex, "role":role, "model":model, "reservation":reserved,
                "purpose":purpose, "context_policy":lower.get("x-autoharness-context-policy"),
                "run_id":lower.get("x-autoharness-run"), "benchmark":lower.get("x-autoharness-benchmark"),
                "task_id":lower.get("x-autoharness-task"), "created":datetime.now(timezone.utc).isoformat(), "status":"reserved"}
            row["received"] = received_at or row["created"]
            row["stream_transport_version"] = STREAM_TRANSPORT_VERSION
            if not alive():
                raise ClientDisconnected("Caller disconnected before dispatch")
            self.append(row)
            self.active_requests.add(row["id"])
        upstream_headers = {"Authorization":"Bearer " + auth["access"], "x-opencode-org-id":auth["metadata"]["orgID"],
            "User-Agent":"autoharness-opencode/1.19.5", "Content-Type":"application/json",
            "x-opencode-session": lower.get("x-opencode-session") or ("autoharness-" + (row["run_id"] or "calibration") + "-" + (row["task_id"] or role))}
        for key in ("anthropic-version", "anthropic-beta"):
            if lower.get(key):
                upstream_headers[key] = lower[key]
        raw = bytearray()
        try:
            with ResponseEvidence(self.artifacts / "response_evidence", row["id"]) as capture:
                row["response_evidence"] = {"body": str(capture.body_path), "metadata": str(capture.meta_path)}
                with self.client.stream("POST", "https://opencode.ai/inference/go" + path, json=payload, headers=upstream_headers) as response:
                    capture.headers(response.status_code, response.headers.get("content-type", "application/json"))
                    if response.status_code != 200:
                        error_raw = response.read()
                        capture.append_before_forward(error_raw)
                        capture.state["transport_eof"] = True
                        save(self.artifacts / "last_error.json", {"status":response.status_code, "body":error_raw.decode(errors="replace")[:4000]})
                        row.update(status="http_error", http_status=response.status_code)
                        raise RuntimeError("Upstream returned HTTP " + str(response.status_code) + "; no automatic retry")
                    response_headers = {"Content-Type":response.headers.get("content-type", "application/json")}
                    streaming = bool(payload.get("stream")) and stream_start is not None and stream_chunk is not None
                    if streaming:
                        stream_start(200, response_headers)
                    pending = bytearray()
                    held = bytearray()
                    ended = False
                    for chunk in capture.chunks(watched_chunks(response,alive,lambda:self.closing)):
                        if "first_chunk_at" not in row:
                            row["first_chunk_at"] = datetime.now(timezone.utc).isoformat()
                        raw.extend(chunk)
                        if not alive():
                            raise ClientDisconnected("Native client disconnected during response")
                        if not streaming:
                            continue
                        pending.extend(chunk)
                        while True:
                            separators = [(pending.find(sep), len(sep)) for sep in (b"\n\n", b"\r\n\r\n") if pending.find(sep) >= 0]
                            if not separators:
                                break
                            pos, width = min(separators)
                            frame = bytes(pending[:pos+width])
                            del pending[:pos+width]
                            ended = ended or terminal_frame(frame)
                            if ended:
                                held.extend(frame)
                            else:
                                stream_chunk(frame)
                    row["usage"] = metered_usage(usage_events(bytes(raw), bool(payload.get("stream"))))
                    if row["usage"]["allowance"] > reserved:
                        raise RuntimeError("Actual allowance exceeded conservative reservation")
            if streaming:
                tail = bytes(held + pending)
                if tail:
                    stream_chunk(tail)
            row["status"] = "complete"
            return 200, response_headers, b"" if streaming else bytes(raw)
        except ClientDisconnected as exc:
            # The model may have completed before the final client write failed.
            # Keep any complete official usage; otherwise retain the full reserve.
            if "usage" not in row:
                try:
                    row["usage"] = metered_usage(usage_events(bytes(raw), bool(payload.get("stream"))))
                except RuntimeError as billing_error:
                    try:
                        events = usage_events(bytes(raw), bool(payload.get("stream")))
                    except (ValueError, UnicodeError):
                        events = []
                    if any("opencode" in (e.get("usage") or (e.get("message") or {}).get("usage") or {}) for e in events):
                        with self.lock:
                            self.halted = str(billing_error)
                except (ValueError, UnicodeError):
                    pass
            if row.get("usage", {}).get("allowance", 0) > reserved:
                with self.lock:
                    self.halted = "Actual allowance exceeded conservative reservation"
            row.update(status="client_disconnected", error=str(exc))
            raise
        except Exception as exc:
            with self.lock:
                self.halted = type(exc).__name__ + ": " + str(exc)
                row["error"] = self.halted
            raise
        finally:
            row["finished"] = datetime.now(timezone.utc).isoformat()
            with self.lock:
                self.append(row)
                self.active_requests.discard(row["id"])
                save(self.artifacts / "status.json", {"requests":len({r["id"] for r in self.rows}), "allowance":self.spent(), "halted":self.halted})

    def __exit__(self, *args):
        with self.lock:
            self.closing = True
        if hasattr(self, "server"):
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(5)
        self.client.close()
        # Closing sockets releases abandoned streams. Let their finally blocks
        # persist unknown reservations before the launcher writes accounting.
        deadline = time.monotonic() + 10
        while self.active_requests and time.monotonic() < deadline:
            time.sleep(0.05)
        if os.environ.get("OPENCODE_GO_API_KEY") == self.token:
            os.environ.pop("OPENCODE_GO_API_KEY", None)
            os.environ.pop("GO_GATEWAY_URL", None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", required=True)
    args = parser.parse_args()
    with httpx.Client(follow_redirects=False) as client:
        report = quota(client, credential())
    save(ROOT / "results/autosaddler_reproduction/go_preflight/quota.json", report)
    print(json.dumps(report, indent=2))
