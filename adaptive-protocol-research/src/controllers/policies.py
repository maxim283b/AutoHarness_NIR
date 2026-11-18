from __future__ import annotations

import json
import math
import random
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from ..execution.state import TaskState, Protocol, PROTOCOLS, Usage, Event
from ..execution.llm import Client


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol: Protocol
    reason: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0, le=1)


def features(state: TaskState, max_tokens: int) -> list[float]:
    prompt = state.prompt.lower()
    return [1.0, min(len(prompt) / 200, 1), float("list" in prompt or "array" in prompt), float("recursive" in prompt or "tree" in prompt), float("number" in prompt or "math" in prompt), min(state.failed_attempts / 2, 1), min(state.attempt / 2, 1), max(0.0, 1.0 - state.usage.tokens / max_tokens)]


class Fixed:
    def __init__(self, protocol: Protocol):
        self.protocol = protocol

    def choose(self, state: TaskState, client: Client) -> Decision:
        return Decision(protocol=self.protocol, reason="fixed control", confidence=1)


class Rule:
    def choose(self, state: TaskState, client: Client) -> Decision:
        if state.attempt == 0:
            choice: Protocol = "P2" if len(state.prompt) > 150 else "P1"
            reason = "long prompt" if choice == "P2" else "short prompt"
        elif state.failed_attempts >= 2:
            choice, reason = "P4", "two failed visible checks"
        elif state.failed_attempts:
            choice, reason = "P3", "visible check failed"
        else:
            choice, reason = "P1", "visible check passed"
        return Decision(protocol=choice, reason=reason, confidence=0.7)


class LLMController:
    def choose(self, state: TaskState, client: Client) -> Decision:
        prompt = f"Choose one protocol for the next implementation stage: P1 single executor, P2 planner-executor, P3 executor-verifier, P4 planner-executor-verifier. Return only JSON with exactly protocol, reason, confidence (number 0..1). No benchmark answers. Task: {state.prompt}\nAttempt: {state.attempt}; failures: {state.failed_attempts}; visible feedback: {state.feedback}; tokens used: {state.usage.tokens}; prior plan: {state.plan[:500]}; prior critique: {state.critique[:500]}"
        raw, usage = client.ask("controller", prompt)
        state.usage.add(usage)
        state.controller_usage.add(usage)
        state.events.append(Event(kind="controller", attempt=state.attempt, protocol=state.protocol, content=raw, usage=usage))
        try:
            value = json.loads(raw)
            return Decision.model_validate(value)
        except (ValueError, ValidationError):
            return Decision(protocol=state.protocol, reason="invalid controller JSON; retain protocol", confidence=0)


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    a = [row[:] + [v] for row, v in zip(matrix, vector)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(a[i][col]))
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= scale
        for i in range(n):
            if i != col:
                factor = a[i][col]
                for j in range(col, n + 1):
                    a[i][j] -= factor * a[col][j]
    return [row[-1] for row in a]


class LinUCB:
    """Small disjoint-arm ridge UCB; final task reward is assigned to selected stages."""
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.dim = 8
        self.a = {p: [[float(i == j) for j in range(self.dim)] for i in range(self.dim)] for p in PROTOCOLS}
        self.b = {p: [0.0] * self.dim for p in PROTOCOLS}
        self.observations = 0

    def update(self, protocol: Protocol, x: list[float], reward: float) -> None:
        for i in range(self.dim):
            self.b[protocol][i] += reward * x[i]
            for j in range(self.dim):
                self.a[protocol][i][j] += x[i] * x[j]
        self.observations += 1

    def choose(self, state: TaskState, client: Client) -> Decision:
        x = features(state, client.limits.max_tokens)
        scores = {}
        for p in PROTOCOLS:
            theta = solve_linear(self.a[p], self.b[p])
            inv_x = solve_linear(self.a[p], x)
            scores[p] = sum(u * v for u, v in zip(theta, x)) + self.alpha * math.sqrt(max(0, sum(u * v for u, v in zip(x, inv_x))))
        choice = max(PROTOCOLS, key=lambda p: scores[p])
        return Decision(protocol=choice, reason=f"LinUCB score {scores[choice]:.3f}; n={self.observations}", confidence=0.5)


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def choose(self, state: TaskState, client: Client) -> Decision:
        rng = random.Random(f"{self.seed}:{state.task_id}:{state.attempt}")
        return Decision(protocol=rng.choice(PROTOCOLS), reason="seeded random control", confidence=0.25)
