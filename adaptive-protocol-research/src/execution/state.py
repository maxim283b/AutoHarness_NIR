from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

Protocol = Literal["P1", "P2", "P3", "P4"]
PROTOCOLS: tuple[Protocol, ...] = ("P1", "P2", "P3", "P4")


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    latency_s: float = 0.0
    estimated_tokens: bool = False

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls
        self.latency_s += other.latency_s
        self.estimated_tokens |= other.estimated_tokens


class Event(BaseModel):
    kind: str
    attempt: int
    protocol: Protocol
    content: str = ""
    usage: Usage = Field(default_factory=Usage)


class Switch(BaseModel):
    previous: Protocol
    current: Protocol
    reason: str
    confidence: float | None = None
    attempt: int
    state_snapshot: dict
    tokens_at_switch: int
    latency_at_switch_s: float
    final_success: bool | None = None


class TaskState(BaseModel):
    task_id: int
    prompt: str
    visible_test: str
    imports: list[str]
    protocol: Protocol = "P1"
    attempt: int = 0
    code: str = ""
    plan: str = ""
    critique: str = ""
    feedback: str = ""
    visible_pass: bool = False
    failed_attempts: int = 0
    events: list[Event] = Field(default_factory=list)
    switches: list[Switch] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    controller_usage: Usage = Field(default_factory=Usage)

    def snapshot(self) -> dict:
        return {
            "task_id": self.task_id,
            "attempt": self.attempt,
            "code": self.code,
            "plan": self.plan,
            "critique": self.critique,
            "feedback": self.feedback,
            "visible_pass": self.visible_pass,
            "failed_attempts": self.failed_attempts,
            "event_count": len(self.events),
            "event_history": [event.model_dump() for event in self.events],
        }
