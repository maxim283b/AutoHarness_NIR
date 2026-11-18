"""MBPPHarnessAdapter: the only integration point between our harness and GEPA.

Harness (fixed structure, pilot_selection.md 3.1):
    1. SOLVE   system=solver_prompt, user=task + first assert -> first ```python``` block
    2. CHECK   code + FIRST (visible) assert in subprocess
    3. REPAIR  if visible failed: system=repair_prompt, user=task + code + error -> new code (once)
    4. SCORE   final code + ALL asserts -> 1.0 / 0.0

Only the texts of `solver_prompt` and `repair_prompt` are candidate components.
The GEPA engine is used as-is; this module only implements the GEPAAdapter protocol.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from sandbox import DEFAULT_TIMEOUT_S, run_code_with_asserts

COMPONENTS = ("solver_prompt", "repair_prompt")


@dataclass
class HarnessConfig:
    task_lm: str = "ollama_chat/qwen2.5-coder:1.5b"
    api_base: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 512
    num_ctx: int = 4096
    seed: int = 0
    use_repair: bool = True
    max_repairs: int = 1
    sandbox_timeout_s: float = DEFAULT_TIMEOUT_S
    request_timeout_s: float = 600.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def candidate_hash(candidate: dict[str, str]) -> str:
    # same definition as gepa.core.state._candidate_hash
    return hashlib.sha256(json.dumps(sorted(candidate.items())).encode()).hexdigest()


_FENCE_PY = re.compile(r"```(?:python|py|Python)[ \t]*\r?\n(.*?)```", re.DOTALL)
_FENCE_ANY = re.compile(r"```[^\n]*\r?\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    m = _FENCE_PY.search(text) or _FENCE_ANY.search(text)
    if m:
        return m.group(1).strip("\n")
    # unterminated fence (e.g. truncated by max_tokens)
    m = re.search(r"```(?:python|py)?[ \t]*\r?\n(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip("\n")
    return text.strip()


def solve_user_message(task: dict) -> str:
    return f"{task['prompt']}\nYour code should pass this test:\n{task['test_list'][0]}"


def repair_user_message(task: dict, code: str, error: str) -> str:
    return (
        f"Task:\n{solve_user_message(task)}\n\n"
        f"Previous solution:\n```python\n{code}\n```\n\n"
        f"Test result:\n{error}"
    )


def visible_error_text(task: dict, err: str | None) -> str:
    return f"Test `{task['test_list'][0]}` failed with: {err}"


@dataclass
class _Run:
    run_id: str
    phase: str
    split: str
    log_path: Path | None
    lock: threading.Lock = field(default_factory=threading.Lock)


class MBPPHarnessAdapter(GEPAAdapter):
    def __init__(
        self,
        config: HarnessConfig | None = None,
        log_path: str | Path | None = None,
        run_id: str = "adhoc",
        phase: str = "optimize",
        split: str = "unknown",
        completion_fn=None,
    ):
        self.config = config or HarnessConfig()
        self._run = _Run(run_id, phase, split, Path(log_path) if log_path else None)
        if completion_fn is None:
            import litellm

            completion_fn = litellm.completion
        self._completion = completion_fn

    # ------------------------------------------------------------------ LM
    def _call_lm(self, system: str, user: str) -> dict:
        cfg = self.config
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        t0 = time.perf_counter()
        resp = self._completion(
            model=cfg.task_lm,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
            num_ctx=cfg.num_ctx,
            api_base=cfg.api_base,
            timeout=cfg.request_timeout_s,
        )
        latency = time.perf_counter() - t0
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return {
            "messages": messages,
            "raw_response": choice.message.content or "",
            "finish_reason": getattr(choice, "finish_reason", None),
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "latency_s": latency,
        }

    # --------------------------------------------------------------- logging
    def _log(self, rec: dict) -> None:
        if self._run.log_path is None:
            return
        with self._run.lock, self._run.log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --------------------------------------------------------------- harness
    def run_task(self, task: dict, candidate: dict[str, str]) -> dict:
        cfg = self.config
        cand_h = candidate_hash(candidate)
        base = {
            "run_id": self._run.run_id,
            "phase": self._run.phase,
            "split": self._run.split,
            "task_id": task["task_id"],
            "candidate_hash": cand_h,
            "solver_prompt_hash": sha256_text(candidate["solver_prompt"]),
            "repair_prompt_hash": sha256_text(candidate["repair_prompt"]),
            "model_id": cfg.task_lm,
            "seed": cfg.seed,
        }
        steps: list[dict] = []
        visible = [task["test_list"][0]]

        def step(kind: str, system: str, user: str) -> tuple[str, Any]:
            out = self._call_lm(system, user)
            code = extract_code(out["raw_response"])
            vis = run_code_with_asserts(code, task["test_imports"], visible, cfg.sandbox_timeout_s)
            rec = {
                **base,
                "step": kind,
                **out,
                "extracted_code": code,
                "visible_test_ok": vis.ok,
                "visible_error": None if vis.ok else visible_error_text(task, vis.error),
            }
            steps.append(rec)
            return code, vis

        error = None
        try:
            code, vis = step("solve", candidate["solver_prompt"], solve_user_message(task))
            repairs = 0
            while cfg.use_repair and not vis.ok and repairs < cfg.max_repairs:
                code, vis = step(
                    "repair",
                    candidate["repair_prompt"],
                    repair_user_message(task, code, visible_error_text(task, vis.error)),
                )
                repairs += 1
            final = run_code_with_asserts(code, task["test_imports"], task["test_list"], cfg.sandbox_timeout_s)
        except Exception as e:  # never raise for a single example (GEPA contract)
            error = f"{type(e).__name__}: {e}"
            code = steps[-1]["extracted_code"] if steps else ""
            final = None

        verifier = (
            final.to_dict()
            if final is not None
            else {"ok": False, "n_passed": 0, "n_total": len(task["test_list"]), "error": error, "timed_out": False}
        )
        ts = datetime.now(timezone.utc).isoformat()
        for rec in steps:
            rec.update(
                final_pass=verifier["ok"], n_asserts_passed=verifier["n_passed"], timestamp=ts
            )
            self._log(rec)
        if not steps:
            self._log({**base, "step": "error", "error": error, "final_pass": False, "timestamp": ts})

        return {
            "task_id": task["task_id"],
            "answer": code,
            "score": 1.0 if verifier["ok"] else 0.0,
            "verifier": verifier,
            "repair_used": any(s["step"] == "repair" for s in steps),
            "visible_ok_first": steps[0]["visible_test_ok"] if steps else False,
            "prompt_tokens": sum(s["prompt_tokens"] for s in steps),
            "completion_tokens": sum(s["completion_tokens"] for s in steps),
            "latency_s": sum(s["latency_s"] for s in steps),
            "n_lm_calls": len(steps),
            "seed": cfg.seed,
            "candidate_hash": cand_h,
            "harness_error": error,
            "steps": steps,
        }

    # ------------------------------------------------------------ GEPA API
    def evaluate(self, batch: list[dict], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        results = [self.run_task(t, candidate) for t in batch]
        return EvaluationBatch(
            outputs=results,
            scores=[r["score"] for r in results],
            trajectories=[{"task": t, "result": r} for t, r in zip(batch, results)] if capture_traces else None,
            objective_scores=[
                {
                    "pass": r["score"],
                    "repair_used": float(r["repair_used"]),
                    "tokens": float(r["prompt_tokens"] + r["completion_tokens"]),
                }
                for r in results
            ],
        )

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: EvaluationBatch, components_to_update: list[str]
    ) -> dict[str, list[dict]]:
        # Only the visible (first) assert and the final pass/fail are exposed; hidden asserts never are.
        out: dict[str, list[dict]] = {c: [] for c in components_to_update}
        all_recs: list[dict] = []
        for traj in eval_batch.trajectories or []:
            task, r = traj["task"], traj["result"]
            steps = r["steps"]
            solve = next((s for s in steps if s["step"] == "solve"), None)
            repair = next((s for s in steps if s["step"] == "repair"), None)
            passed = r["verifier"]["ok"]
            feedback = "Final result: PASSED all tests." if passed else "Final result: FAILED (hidden tests not shown)."
            if solve is not None:
                feedback += " Visible test after solve: " + ("passed." if solve["visible_test_ok"] else solve["visible_error"])
            if repair is not None:
                feedback += " Visible test after repair: " + ("passed." if repair["visible_test_ok"] else repair["visible_error"])
            if r["harness_error"]:
                feedback += f" Harness error: {r['harness_error']}"
            rec = {
                "Inputs": {"task": solve_user_message(task)},
                "Generated code": solve["extracted_code"] if solve else "",
                "Visible test result": "passed" if solve and solve["visible_test_ok"] else (solve or {}).get("visible_error"),
                "Repair attempt": repair["extracted_code"] if repair else "(not triggered)",
                "Feedback": feedback,
            }
            all_recs.append(rec)
            if "solver_prompt" in out:
                out["solver_prompt"].append(rec)
            if "repair_prompt" in out and repair is not None:
                out["repair_prompt"].append(rec)
        # no repair was triggered in this minibatch: give the reflector the solve records instead of nothing
        if "repair_prompt" in out and not out["repair_prompt"]:
            out["repair_prompt"] = all_recs
        return out
