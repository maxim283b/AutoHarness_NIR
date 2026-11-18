"""Unified task runner for the four pilot harness patterns."""

from __future__ import annotations

import hashlib
import inspect
import time
from typing import Any, Callable

from .adapters import PythonTaskAdapter
from .schema import HarnessConfig, Step, ToolResult, Trajectory


def _source_hash(fn: Callable) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def run_task(task, harness_config: HarnessConfig, model_client, verifier, memory=None, adapter=None) -> Trajectory:
    """Run one task. `verifier` is injected and is never changed by the runner.

    Accepted verifier return values are bool, ``(ok, error)``, or a mapping with
    ``success``/``error`` keys. The model client must expose ``chat(messages)``.
    """
    adapter = adapter or PythonTaskAdapter()
    started = time.perf_counter()
    source_hash = harness_config.harness_source_hash or _source_hash(run_task)
    tr = Trajectory("1.1", task.get("task_id"), adapter.public_task(task), harness_config.to_dict(),
                    harness_config.config_hash, source_hash, harness_config.model.__dict__,
                    harness_config.seed, timestamp=time.time())
    idx = 0

    def model(name, system, user):
        nonlocal idx
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            r = model_client.chat(messages)
            raw = r.get("text") or ""
            s = Step(idx, "model", name, input=user, output=adapter.extract_output(raw), messages=messages,
                     prompt_tokens=int(r.get("prompt_tokens", 0)), completion_tokens=int(r.get("completion_tokens", 0)),
                     calls=1, latency_s=float(r.get("latency_s", 0.0)))
        except Exception as exc:
            s = Step(idx, "model", name, input=user, error_type=type(exc).__name__, error=str(exc), calls=1)
            tr.error_type = type(exc).__name__
        tr.steps.append(s); idx += 1
        return s.output or ""

    def tool(name, inp, fn):
        nonlocal idx
        t0 = time.perf_counter()
        try:
            out = fn(inp)
            ok, err = _verifier_result(out)
            result = ToolResult(name, inp, out, ok, err, time.time(), time.perf_counter() - t0)
            s = Step(idx, "verifier" if name == "verifier" else "tool", name, input=inp, output=out,
                     tool_result=result, calls=1, latency_s=result.duration_s or 0.0)
        except Exception as exc:
            result = ToolResult(name, inp, None, False, str(exc), time.time(), time.perf_counter() - t0)
            s = Step(idx, "verifier" if name == "verifier" else "tool", name, input=inp,
                     tool_result=result, calls=1, latency_s=result.duration_s or 0.0,
                     error_type=type(exc).__name__, error=str(exc))
        tr.steps.append(s); idx += 1
        return result

    system = harness_config.system_prompt
    examples = memory.retrieve(task.get("prompt", task.get("description", ""))) if harness_config.pattern == "memory_assisted" and memory is not None else None
    output = model("solve_with_memory" if examples else "solve", system,
                   adapter.initial_user(task, harness_config, examples))
    final = output
    last_vr = None
    if harness_config.pattern == "self_refine":
        final = model("refine", system, adapter.refine_user(task, output))
    elif harness_config.pattern == "verifier_loop":
        for repair in range(harness_config.max_repairs + 1):
            vr = tool("verifier", final, verifier)
            last_vr = vr
            if vr.ok or repair == harness_config.max_repairs:
                break
            final = model(f"repair_{repair + 1}", system, adapter.repair_user(task, final, vr.error))
    # The loop already verified the final candidate. Avoid a duplicate call;
    # this also preserves a strict one-verifier-per-attempt accounting.
    vr = last_vr or tool("verifier", final, verifier)
    tr.final_output = final
    tr.success = bool(vr.ok)
    tr.stop_reason = "success" if tr.success else ("max_repairs" if harness_config.pattern == "verifier_loop" else "final_verifier")
    tr.total_prompt_tokens = sum(s.prompt_tokens for s in tr.steps)
    tr.total_completion_tokens = sum(s.completion_tokens for s in tr.steps)
    tr.model_calls = sum(s.calls for s in tr.steps if s.kind == "model")
    tr.tool_calls = sum(s.calls for s in tr.steps if s.kind in {"tool", "verifier"})
    tr.latency_s = sum(s.latency_s for s in tr.steps)
    tr.wall_time_s = time.perf_counter() - started
    return tr


def _verifier_result(value):
    if isinstance(value, dict):
        return bool(value.get("success", value.get("ok", False))), value.get("error")
    if isinstance(value, tuple):
        return bool(value[0]), value[1] if len(value) > 1 else None
    return bool(value), None
