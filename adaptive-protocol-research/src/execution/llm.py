from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from .state import Usage


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Limits:
    max_calls: int = 8
    max_tokens: int = 10000
    max_seconds: float = 120.0


class Client:
    def __init__(self, model: str, base_url: str, key: str, limits: Limits, mock: bool = False):
        self.model, self.base_url, self.key = model, base_url.rstrip("/"), key
        self.limits, self.mock = limits, mock
        self.usage = Usage()
        self.started = time.monotonic()

    @classmethod
    def from_env(cls, limits: Limits, mock: bool = False) -> "Client":
        model = os.getenv("OPENAI_MODEL", "")
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        key = os.getenv("OPENAI_API_KEY", "")
        if not mock and (not model or not key):
            raise ValueError("Set OPENAI_MODEL and OPENAI_API_KEY for a real run")
        return cls(model or "scripted-mock", base, key, limits, mock)

    def ask(self, role: str, prompt: str) -> tuple[str, Usage]:
        if self.usage.calls >= self.limits.max_calls or self.usage.tokens >= self.limits.max_tokens:
            raise BudgetExceeded("call/token limit")
        if time.monotonic() - self.started >= self.limits.max_seconds:
            raise BudgetExceeded("wall-time limit")
        start = time.monotonic()
        if self.mock:
            if role == "controller":
                content = '{"protocol":"P3","reason":"mock decision","confidence":0.5}'
            elif role == "planner":
                content = "Implement the requested function and check the example."
            elif role == "verifier":
                content = "Check the failing assertion and repair the implementation."
            else:
                match = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", prompt)
                name = match.group(1) if match else "solution"
                content = f"def {name}(*args, **kwargs):\n    return None"
            usage = Usage(prompt_tokens=max(1, len(prompt) // 4), completion_tokens=max(1, len(content) // 4), calls=1, latency_s=time.monotonic() - start, estimated_tokens=True)
        else:
            role_caps = {"planner": 80, "verifier": 96, "controller": 64, "executor": 300}
            payload = {"model": self.model, "messages": [{"role": "system", "content": f"You are the {role} role in a Python programming task. Follow the user message exactly."}, {"role": "user", "content": prompt}], "temperature": 0, "max_tokens": role_caps.get(role, 400)}
            req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
            timeout = max(1.0, self.limits.max_seconds - (time.monotonic() - self.started))
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.load(response)
            content = data["choices"][0]["message"].get("content") or ""
            raw = data.get("usage") or {}
            estimated = not ("prompt_tokens" in raw and "completion_tokens" in raw)
            usage = Usage(prompt_tokens=int(raw.get("prompt_tokens", max(1, len(prompt) // 4))), completion_tokens=int(raw.get("completion_tokens", max(1, len(content) // 4))), calls=1, latency_s=time.monotonic() - start, estimated_tokens=estimated)
        self.usage.add(usage)
        if self.usage.tokens > self.limits.max_tokens or time.monotonic() - self.started > self.limits.max_seconds:
            raise BudgetExceeded("token/wall-time limit exceeded after response")
        return content, usage
