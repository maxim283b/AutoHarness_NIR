from __future__ import annotations

import os
import time
from typing import Protocol as TypingProtocol
from .state import TaskState, Switch, Protocol
from .llm import BudgetExceeded, Client
from ..protocols.roles import execute_stage
from ..evaluation.checker import run_tests


class Policy(TypingProtocol):
    def choose(self, state: TaskState, client: Client): ...


def execute_task(task: dict, policy: Policy, client: Client, *, max_attempts: int = 3, max_switches: int = 2, front_only: bool = False) -> dict:
    started = time.monotonic()
    state = TaskState(task_id=task["task_id"], prompt=task["prompt"], visible_test=task["test_list"][0], imports=task.get("test_imports", []))
    error = ""
    initial_protocol: Protocol = state.protocol
    try:
        for _ in range(max_attempts):
            if state.attempt == 0 or not front_only:
                decision = policy.choose(state, client)
                choice: Protocol = decision.protocol
                if state.attempt == 0:
                    state.protocol = choice
                    initial_protocol = choice
                elif choice != state.protocol and len(state.switches) < max_switches:
                    state.switches.append(Switch(previous=state.protocol, current=choice, reason=decision.reason, confidence=decision.confidence, attempt=state.attempt, state_snapshot=state.snapshot(), tokens_at_switch=state.usage.tokens, latency_at_switch_s=time.monotonic() - started))
                    state.protocol = choice
            execute_stage(state, client, state.protocol)
            if state.visible_pass:
                break
    except (BudgetExceeded, TimeoutError, OSError, ValueError, KeyError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    hidden = task["test_list"][1:]
    hidden_ok, hidden_error = run_tests(state.code, state.imports, hidden) if state.code else (False, "no code")
    success = state.visible_pass and hidden_ok
    # The final score is never placed in prompts, state, or controller memory during this task.
    for switch in state.switches:
        switch.final_success = success
    input_rate = os.getenv("OPENAI_INPUT_USD_PER_MTOK")
    output_rate = os.getenv("OPENAI_OUTPUT_USD_PER_MTOK")
    cost = (state.usage.prompt_tokens * float(input_rate) + state.usage.completion_tokens * float(output_rate)) / 1_000_000 if input_rate and output_rate else None
    controller_cost = (state.controller_usage.prompt_tokens * float(input_rate) + state.controller_usage.completion_tokens * float(output_rate)) / 1_000_000 if input_rate and output_rate else None
    return {
        "task_id": state.task_id, "success": success, "hidden_error": hidden_error, "error": error,
        "initial_protocol": initial_protocol,
        "final_protocol": state.protocol, "attempts": state.attempt, "switch_count": len(state.switches),
        "errors": state.failed_attempts + int(bool(error)), "tool_calls": sum(event.kind == "visible_check" for event in state.events),
        "switches": [s.model_dump() for s in state.switches], "events": [e.model_dump() for e in state.events],
        "state": state.snapshot(), "prompt_tokens": state.usage.prompt_tokens, "completion_tokens": state.usage.completion_tokens,
        "tokens": state.usage.tokens, "calls": state.usage.calls, "controller_tokens": state.controller_usage.tokens,
        "controller_calls": state.controller_usage.calls, "controller_latency_s": state.controller_usage.latency_s,
        "llm_latency_s": state.usage.latency_s, "wall_s": time.monotonic() - started, "cost_usd": cost, "controller_cost_usd": controller_cost,
        "estimated_tokens": state.usage.estimated_tokens, "mock": client.mock,
    }
