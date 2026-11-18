"""Component-map code harness: SOLVE -> visible test -> bounded REPAIR -> SCORE.

Components are plain text so AutoSaddler can patch them:
  solver_prompt  - system prompt of the first attempt
  checklist      - extra rules appended to the solver system prompt
  repair_prompt  - system prompt of the repair attempt
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

from harness_lab._paths import ROOT
from harness_lab.llm import OllamaClient

COMPONENTS = ("solver_prompt", "checklist", "repair_prompt")

SEED_COMPONENTS = {
    "solver_prompt": (
        "You are a Python programmer. Write a correct, self-contained Python function that solves "
        "the task. Return only one ```python``` code block."
    ),
    "checklist": "Use the exact function name from the test. Do not print or read input.",
    "repair_prompt": (
        "The previous solution failed a test. Read the error, fix the function, and return only one "
        "corrected ```python``` code block."
    ),
}


def _load_sandbox():
    spec = importlib.util.spec_from_file_location("mbpp_sandbox", ROOT / "pilot_baseline" / "sandbox.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mbpp_sandbox"] = module
    spec.loader.exec_module(module)
    return module


_SANDBOX = _load_sandbox()
run_code_with_asserts = _SANDBOX.run_code_with_asserts

_FENCE = re.compile(r"```(?:python|py|Python)?[ \t]*\r?\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip("\n")
    m = re.search(r"```(?:python|py)?[ \t]*\r?\n(.*)", text, re.DOTALL)
    return (m.group(1) if m else text).strip("\n")


def solve_message(task: dict) -> str:
    visible = task.get("visible_test")
    suffix = f"\nYour code should pass this test:\n{visible}" if visible else ""
    return f"{task['prompt']}{suffix}"


def run_task(task: dict, components: dict[str, str], client: OllamaClient, *,
             use_repair: bool = True, max_repairs: int = 1, timeout_s: float = 10.0) -> dict[str, Any]:
    """Run one task through the harness. `task` needs prompt, imports, tests, visible_test."""
    calls_before = len(client.calls)
    system = components["solver_prompt"].strip()
    if components.get("checklist", "").strip():
        system += "\n\n" + components["checklist"].strip()
    reply = client.chat([{"role": "system", "content": system},
                         {"role": "user", "content": solve_message(task)}])
    code = extract_code(reply["text"])
    first_code = code
    visible_ok = None
    first_error = None
    repairs = 0
    if task.get("visible_test"):
        check = run_code_with_asserts(code, task["imports"], [task["visible_test"]], timeout_s)
        visible_ok, first_error = check.ok, check.error
        while use_repair and not visible_ok and repairs < max_repairs:
            repairs += 1
            msg = (f"Task:\n{solve_message(task)}\n\nPrevious solution:\n```python\n{code}\n```\n\n"
                   f"Test result:\nTest `{task['visible_test']}` failed with: {first_error}")
            reply = client.chat([{"role": "system", "content": components["repair_prompt"].strip()},
                                 {"role": "user", "content": msg}])
            code = extract_code(reply["text"])
            check = run_code_with_asserts(code, task["imports"], [task["visible_test"]], timeout_s)
            visible_ok = check.ok
    final = run_code_with_asserts(code, task["imports"], task["tests"], timeout_s)
    used = client.calls[calls_before:]
    return {
        "task_id": task["task_id"],
        "success": bool(final.ok),
        "reward": 1.0 if final.ok else 0.0,
        "first_pass_ok": first_error is None if task.get("visible_test") else None,
        "repairs": repairs,
        "final_error": final.error,
        "first_error": first_error,
        "first_code": first_code,
        "final_code": code,
        "target_calls": len(used),
        "prompt_tokens": sum(c["prompt_tokens"] for c in used),
        "completion_tokens": sum(c["completion_tokens"] for c in used),
        "tokens": sum(c["prompt_tokens"] + c["completion_tokens"] for c in used),
        "latency_s": sum(c["latency_s"] for c in used),
    }
