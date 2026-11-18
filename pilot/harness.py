"""The fixed two-step harness: SOLVE -> CHECK (visible assert) -> one REPAIR -> SCORE (all asserts).

Identical for every arm; only the two prompt texts (candidate) vary.
"""

import hashlib
import json
import os
import re
import threading
import time

from sandbox import run_code_with_asserts

TARGET_MODEL = os.environ.get("PILOT_MODEL", "ministral-3-8b-instruct-2512")
TARGET_OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 4096, "num_predict": 512}
TIMEOUT_S = 10.0

_FENCE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python|py|Python)[ \t]*\n(.*?)```", text, re.S) or _FENCE.search(text)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*)", text, re.S)  # unterminated fence
    return (m.group(1) if m else text).strip()


def sha(obj) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def solve_user_msg(task: dict) -> str:
    return f"{task['prompt']}\nYour code should pass this test:\n{task['test_list'][0]}"


def repair_user_msg(task: dict, code: str, error: str) -> str:
    return (
        f"Task:\n{task['prompt']}\nYour code should pass this test:\n{task['test_list'][0]}\n\n"
        f"Previous solution:\n```python\n{code}\n```\n\nError:\n{error}"
    )


class TrajectoryLog:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, rec: dict):
        if self.path is None:
            return
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_task(task: dict, candidate: dict, client, log: TrajectoryLog | None, ctx: dict, use_repair: bool = True) -> dict:
    """Run the harness on one task. Never raises for model/code failures. Returns a trajectory dict."""
    cand_hash = sha(candidate)
    base = {
        "run_id": ctx.get("run_id"),
        "arm": ctx.get("arm"),
        "phase": ctx.get("phase"),
        "split": ctx.get("split_of", {}).get(task["task_id"], ctx.get("split")),
        "task_id": task["task_id"],
        "candidate_hash": cand_hash,
        "solver_prompt_hash": sha(candidate["solver_prompt"]),
        "repair_prompt_hash": sha(candidate["repair_prompt"]),
        "model_id": f"{os.environ.get('PILOT_BACKEND', 'lmstudio')}/{TARGET_MODEL}",
    }
    visible = task["test_list"][:1]
    steps = []

    def step(name, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        r = client.chat(msgs)
        code = extract_code(r["text"])
        ok, _, err, _ = run_code_with_asserts(code, task["test_imports"], visible, TIMEOUT_S)
        s = {"step": name, "messages": msgs, "raw_response": r["text"], "extracted_code": code,
             "visible_test_ok": ok, "visible_error": err, "prompt_tokens": r["prompt_tokens"],
             "completion_tokens": r["completion_tokens"], "latency_s": r["latency_s"]}
        steps.append(s)
        return s

    s1 = step("solve", candidate["solver_prompt"], solve_user_msg(task))
    final = s1
    if use_repair and not s1["visible_test_ok"]:
        final = step("repair", candidate["repair_prompt"],
                     repair_user_msg(task, s1["extracted_code"], s1["visible_error"] or "unknown error"))
    ok, n_passed, _, _ = run_code_with_asserts(final["extracted_code"], task["test_imports"], task["test_list"], TIMEOUT_S)
    for s in steps:
        if log is not None:
            log.write({**base, **s, "final_pass": ok, "n_asserts_passed": n_passed,
                       "n_asserts": len(task["test_list"]), "timestamp": time.time()})
    return {
        "task_id": task["task_id"],
        "task": task,
        "candidate_hash": cand_hash,
        "steps": steps,
        "repair_used": len(steps) > 1,
        "final_code": final["extracted_code"],
        "final_pass": ok,
        "n_asserts_passed": n_passed,
        "tokens": sum(s["prompt_tokens"] + s["completion_tokens"] for s in steps),
    }
