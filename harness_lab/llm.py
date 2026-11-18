"""Minimal Ollama client (native /api/chat) with token accounting and call recording."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMSettings:
    model: str = "qwen2.5:7b"
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    seed: int = 0
    num_ctx: int = 4096
    max_tokens: int = 512
    timeout_s: float = 600.0
    keep_alive: str = "30m"
    api_key_env: str = ""  # name of the env var holding the API key (OpenAI-compatible / Anthropic)


@dataclass
class OllamaClient:
    settings: LLMSettings = field(default_factory=LLMSettings)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int | None = None,
             json_format: bool | dict | None = None, temperature: float | None = None) -> dict[str, Any]:
        s = self.settings
        if "anthropic.com" in s.base_url:
            return self._chat_anthropic(messages, max_tokens=max_tokens, temperature=temperature)
        if s.base_url.rstrip("/").endswith("/v1"):
            return self._chat_openai(messages, max_tokens=max_tokens, json_format=json_format,
                                     temperature=temperature)
        body: dict[str, Any] = {
            "model": s.model,
            "messages": messages,
            "stream": False,
            "keep_alive": s.keep_alive,
            "options": {
                "temperature": s.temperature if temperature is None else temperature,
                "seed": s.seed,
                "num_ctx": s.num_ctx,
                "num_predict": max_tokens or s.max_tokens,
            },
        }
        if json_format:
            body["format"] = json_format
        request = urllib.request.Request(
            s.base_url.rstrip("/") + "/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=s.timeout_s) as response:
            out = json.loads(response.read())
        elapsed = time.perf_counter() - started
        record = {
            "text": (out.get("message") or {}).get("content", "").strip(),
            "prompt_tokens": int(out.get("prompt_eval_count", 0)),
            "completion_tokens": int(out.get("eval_count", 0)),
            "latency_s": elapsed,
            "done_reason": out.get("done_reason"),
        }
        self.calls.append(record)
        return record

    def _chat_openai(self, messages: list[dict[str, str]], *, max_tokens: int | None,
                     json_format: bool | dict | None, temperature: float | None) -> dict[str, Any]:
        """OpenAI-compatible /v1/chat/completions (LM Studio)."""
        s = self.settings
        body: dict[str, Any] = {
            "model": s.model,
            "messages": messages,
            "stream": False,
            "temperature": s.temperature if temperature is None else temperature,
            "seed": s.seed,
            "max_tokens": max_tokens or s.max_tokens,
        }
        if json_format:
            body["response_format"] = ({"type": "json_object"} if json_format is True else json_format)
        request = urllib.request.Request(
            s.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + (os.environ.get(s.api_key_env, "") if s.api_key_env else "lm-studio")},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=s.timeout_s) as response:
            out = json.loads(response.read())
        elapsed = time.perf_counter() - started
        choice = (out.get("choices") or [{}])[0]
        usage = out.get("usage") or {}
        record = {
            "text": ((choice.get("message") or {}).get("content") or "").strip(),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "latency_s": elapsed,
            "done_reason": choice.get("finish_reason"),
        }
        self.calls.append(record)
        return record

    def _chat_anthropic(self, messages: list[dict[str, str]], *, max_tokens: int | None,
                        temperature: float | None) -> dict[str, Any]:
        """Anthropic Messages API (base_url e.g. https://api.anthropic.com/v1)."""
        s = self.settings
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        body: dict[str, Any] = {
            "model": s.model,
            "max_tokens": max_tokens or s.max_tokens,
            "temperature": s.temperature if temperature is None else temperature,
            "messages": [m for m in messages if m["role"] != "system"],
        }
        if system:
            body["system"] = system
        request = urllib.request.Request(
            s.base_url.rstrip("/") + "/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01",
                     "x-api-key": os.environ.get(s.api_key_env or "ANTHROPIC_API_KEY", "")},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=s.timeout_s) as response:
            out = json.loads(response.read())
        elapsed = time.perf_counter() - started
        usage = out.get("usage") or {}
        record = {
            "text": "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text").strip(),
            "prompt_tokens": int(usage.get("input_tokens", 0)),
            "completion_tokens": int(usage.get("output_tokens", 0)),
            "latency_s": elapsed,
            "done_reason": out.get("stop_reason"),
        }
        self.calls.append(record)
        return record

    @property
    def total_tokens(self) -> int:
        return sum(c["prompt_tokens"] + c["completion_tokens"] for c in self.calls)
