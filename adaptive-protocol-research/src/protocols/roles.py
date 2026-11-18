from __future__ import annotations

import re
from ..execution.state import TaskState, Event, Protocol
from ..execution.llm import Client
from ..evaluation.checker import run_tests


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S | re.I)
    return (match.group(1) if match else text).strip()


def call(client: Client, state: TaskState, role: str, prompt: str) -> str:
    content, usage = client.ask(role, prompt)
    state.usage.add(usage)
    state.events.append(Event(kind=role, attempt=state.attempt, protocol=state.protocol, content=content, usage=usage))
    return content


def execute_stage(state: TaskState, client: Client, protocol: Protocol) -> None:
    state.protocol = protocol
    checks = [f"attempt {event.attempt + 1}: {event.content}" for event in state.events if event.kind == "visible_check"]
    context = f"Task: {state.prompt}\nVisible example: {state.visible_test}\nImports available: {state.imports}\nAttempt: {state.attempt + 1}\nPrevious code:\n{state.code}\nPrevious plan: {state.plan}\nVisible feedback: {state.feedback}\nTool history: {checks}\nPrevious critique: {state.critique}"
    if protocol in ("P2", "P4"):
        state.plan = call(client, state, "planner", context + "\nWrite a concise implementation plan. Update any prior plan if needed.")
    executor_prompt = context + f"\nPlan: {state.plan if protocol in ('P2','P4') else '(none)'}\nReturn only complete executable Python code. Keep the required function name."
    state.code = extract_code(call(client, state, "executor", executor_prompt))
    state.visible_pass, state.feedback = run_tests(state.code, state.imports, [state.visible_test])
    state.events.append(Event(kind="visible_check", attempt=state.attempt, protocol=protocol, content=f"{state.visible_pass}: {state.feedback}"))
    if not state.visible_pass:
        state.failed_attempts += 1
    if protocol in ("P3", "P4"):
        state.critique = call(client, state, "verifier", context + f"\nCurrent code:\n{state.code}\nVisible check passed: {state.visible_pass}\nVisible error: {state.feedback}\nReview the code for the task, identify a concrete correction if needed. You cannot see hidden tests.")
    state.attempt += 1
