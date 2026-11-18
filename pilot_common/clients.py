"""Small HTTP clients for Ollama and OpenAI-compatible local servers."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any


class _HTTPClient:
    def __init__(self, config):
        self.config = config

    def _post(self, url: str, body: dict[str, Any], headers: dict[str, str] | None = None) -> tuple[dict, float]:
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.config.timeout_s) as response:
            return json.loads(response.read()), time.perf_counter() - t0


class OllamaClient(_HTTPClient):
    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        options = {"temperature": self.config.temperature, "seed": self.config.seed,
                   "num_predict": self.config.max_tokens, **self.config.extra}
        if self.config.num_ctx is not None:
            options["num_ctx"] = self.config.num_ctx
        out, latency = self._post(self.config.base_url.rstrip("/") + "/api/chat",
                                  {"model": self.config.model, "messages": messages,
                                   "stream": False, "options": options})
        return {"text": out["message"]["content"], "prompt_tokens": int(out.get("prompt_eval_count", 0)),
                "completion_tokens": int(out.get("eval_count", 0)), "latency_s": latency,
                "raw": out}


class OpenAICompatibleClient(_HTTPClient):
    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body = {"model": self.config.model, "messages": messages,
                "temperature": self.config.temperature, "max_tokens": self.config.max_tokens,
                "seed": self.config.seed, **self.config.extra}
        key = os.environ.get(self.config.api_key_env, "") if self.config.api_key_env else ""
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        out, latency = self._post(self.config.base_url.rstrip("/") + "/v1/chat/completions", body, headers)
        usage = out.get("usage") or {}
        return {"text": out["choices"][0]["message"]["content"],
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "latency_s": latency, "raw": out}

