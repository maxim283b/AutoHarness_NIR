"""Serializable schemas for harnesses and complete task trajectories.

The module deliberately uses stdlib dataclasses rather than requiring Pydantic,
so it can be imported by the existing pilot environment unchanged.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

Pattern = Literal["direct", "self_refine", "verifier_loop", "memory_assisted"]


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(value: Any) -> str:
    raw = value if isinstance(value, bytes) else canonical_json(value).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ModelConfig:
    backend: Literal["ollama", "openai_compatible"] = "ollama"
    model: str = "qwen2.5-coder:1.5b"
    base_url: str = "http://127.0.0.1:11434"
    api_key_env: str | None = None
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 512
    num_ctx: int | None = 4096
    timeout_s: float = 1800.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessConfig:
    system_prompt: str
    pattern: Pattern
    max_repairs: int = 0
    memory_k: int = 0
    use_visible_verifier: bool = True
    stop_rule: str = "after_final_verifier"
    tool_policy: dict[str, Any] = field(default_factory=dict)
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 0
    harness_source_hash: str = ""

    def __post_init__(self) -> None:
        if self.max_repairs < 0 or self.memory_k < 0:
            raise ValueError("max_repairs and memory_k must be non-negative")
        if self.pattern not in {"direct", "self_refine", "verifier_loop", "memory_assisted"}:
            raise ValueError(f"unsupported pattern: {self.pattern}")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @property
    def config_hash(self) -> str:
        return sha256(self.to_dict())


@dataclass
class ToolResult:
    name: str
    input: Any = None
    output: Any = None
    ok: bool | None = None
    error: str | None = None
    started_at: float | None = None
    duration_s: float | None = None


@dataclass
class Step:
    index: int
    kind: Literal["model", "tool", "verifier"]
    name: str
    input: Any = None
    output: Any = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_result: ToolResult | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    latency_s: float = 0.0
    error_type: str | None = None
    error: str | None = None


@dataclass
class Trajectory:
    schema_version: str
    task_id: Any
    task_input: Any
    harness: dict[str, Any]
    harness_config_hash: str
    harness_source_hash: str
    model: dict[str, Any]
    seed: int
    steps: list[Step] = field(default_factory=list)
    success: bool = False
    error_type: str | None = None
    stop_reason: str | None = None
    final_output: Any = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    latency_s: float = 0.0
    wall_time_s: float = 0.0
    timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

