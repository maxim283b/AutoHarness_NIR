"""Local chat clients with exact token accounting.

LM Studio exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint.
Set ``PILOT_BACKEND=lmstudio`` to use it; Ollama remains available for old
runs and tests.
"""

import json
import threading
import time
import urllib.request
import os
from collections import defaultdict

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"


class TokenLedger:
    """Thread-safe token/call/time counters per role ('target', 'teacher')."""

    def __init__(self):
        self._lock = threading.Lock()
        self.c = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0})

    def add(self, role: str, pt: int, ct: int, sec: float):
        with self._lock:
            r = self.c[role]
            r["calls"] += 1
            r["prompt_tokens"] += pt
            r["completion_tokens"] += ct
            r["seconds"] += sec

    def total(self, role: str) -> int:
        r = self.c[role]
        return r["prompt_tokens"] + r["completion_tokens"]

    def load(self, path):
        import os
        if os.path.exists(path):
            for k, v in json.load(open(path, encoding="utf-8")).items():
                self.c[k].update({kk: v[kk] for kk in ("calls", "prompt_tokens", "completion_tokens", "seconds")})

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, indent=1)

    def snapshot(self) -> dict:
        with self._lock:
            return {k: dict(v, total_tokens=v["prompt_tokens"] + v["completion_tokens"]) for k, v in self.c.items()}


class OllamaChat:
    def __init__(self, model: str, options: dict, ledger: TokenLedger | None = None, role: str = "target",
                 timeout: float = 1800.0):
        self.model = model
        self.options = options
        self.ledger = ledger
        self.role = role
        self.timeout = timeout

    def chat(self, messages: list[dict]) -> dict:
        body = json.dumps({"model": self.model, "messages": messages, "stream": False, "options": self.options})
        last_err = None
        for attempt in range(3):
            t0 = time.time()
            try:
                req = urllib.request.Request(OLLAMA_URL, data=body.encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read())
                break
            except Exception as e:  # transient server hiccup: retry, then raise (systemic)
                last_err = e
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"Ollama call failed for {self.model}: {last_err}")
        dt = time.time() - t0
        pt, ct = int(out.get("prompt_eval_count", 0)), int(out.get("eval_count", 0))
        if self.ledger is not None:
            self.ledger.add(self.role, pt, ct, dt)
        return {
            "text": out["message"]["content"],
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "latency_s": round(dt, 3),
            "done_reason": out.get("done_reason"),
        }


class OpenAICompatibleChat:
    """Client for LM Studio and other local OpenAI-compatible servers."""

    def __init__(self, model: str, options: dict, ledger: TokenLedger | None = None, role: str = "target",
                 base_url: str = LM_STUDIO_URL, timeout: float = 1800.0):
        self.model = model
        self.options = options
        self.ledger = ledger
        self.role = role
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, messages: list[dict]) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.options.get("temperature", 0),
            "max_tokens": self.options.get("num_predict", self.options.get("max_tokens", 512)),
            "seed": self.options.get("seed", 0),
            "stream": False,
        }
        if "top_p" in self.options:
            body["top_p"] = self.options["top_p"]
        last_err = None
        for attempt in range(3):
            t0 = time.time()
            try:
                req = urllib.request.Request(
                    self.base_url,
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read())
                break
            except Exception as e:
                last_err = e
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"OpenAI-compatible call failed for {self.model} at {self.base_url}: {last_err}")
        dt = time.time() - t0
        usage = out.get("usage") or {}
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        if self.ledger is not None:
            self.ledger.add(self.role, pt, ct, dt)
        choice = (out.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return {
            "text": message.get("content", ""),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "latency_s": round(dt, 3),
            "done_reason": choice.get("finish_reason"),
        }


def make_chat_client(model: str, options: dict, ledger: TokenLedger | None = None,
                     role: str = "target"):
    """Build the configured local client without changing the harness."""
    backend = os.environ.get("PILOT_BACKEND", "lmstudio").lower()
    if backend in {"lmstudio", "openai_compatible", "openai-compatible"}:
        url = os.environ.get("LM_STUDIO_CHAT_URL", LM_STUDIO_URL)
        return OpenAICompatibleChat(model, options, ledger, role, base_url=url)
    if backend == "ollama":
        return OllamaChat(model, options, ledger, role)
    raise ValueError(f"Unsupported PILOT_BACKEND={backend!r}; use lmstudio or ollama")


class TeacherLM:
    """GEPA LanguageModel: callable(prompt) -> str. Logs every reflection call to a JSONL file."""

    def __init__(self, client: OllamaChat, log_path=None):
        self.client = client
        self.log_path = log_path
        self._lock = threading.Lock()

    def __call__(self, prompt):
        from gepa.lm import LMOutput

        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        r = self.client.chat(messages)
        if self.log_path:
            with self._lock, open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "messages": messages, **r}, ensure_ascii=False) + "\n")
        finish = "length" if r["done_reason"] == "length" else "stop"
        return LMOutput(r["text"], finish_reason=finish)
