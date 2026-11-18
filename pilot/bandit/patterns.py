"""Five fixed harness patterns around the same frozen target model and the same verifier.

Every pattern returns an `Outcome` with the verifier result on ALL asserts (success), the exact token
count of every LM call (Ollama prompt_eval_count + eval_count), the number of tool calls (sandbox
executions of the visible test, memory retrievals) and LM latency. Target weights are never touched:
the only thing that varies between arms is the control flow / prompts around the same model.
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import extract_code, solve_user_msg  # noqa: E402
from sandbox import run_code_with_asserts  # noqa: E402

PATTERNS = ["direct", "plan_execute", "self_refine", "verifier_loop", "memory"]
TIMEOUT_S = 10.0
MAX_REPAIRS = 2  # verifier_loop: solve + up to 2 repairs

SOLVER = ("You are a Python programmer. Write a correct, self-contained Python function that solves the task. "
          "Return only one ```python``` code block.")
PLANNER = ("You are a Python programmer. Write a short numbered plan (at most 5 steps) for implementing a function "
           "that solves the task: name the function exactly as in the test, list inputs, outputs and edge cases. "
           "Do not write any code.")
CRITIC = ("You are a careful Python reviewer. Check the solution against the task and the example test: function "
          "name, argument order, return type, edge cases. Briefly list any problems, then return the complete "
          "corrected function (or the same function if it is already correct) in one ```python``` code block.")
REPAIR = ("The previous solution failed a test. Read the error, fix the function, and return only one corrected "
          "```python``` code block.")
MEMORY = ("You are a Python programmer. Below are solved examples of similar tasks from memory. Use them only as "
          "style and API hints. Write a correct, self-contained Python function that solves the NEW task. "
          "Return only one ```python``` code block.")


class Recorder:
    def __init__(self, client):
        self.client = client
        self.steps = []
        self.tool_calls = 0

    def lm(self, name, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        r = self.client.chat(msgs)
        self.steps.append({"step": name, "messages": msgs, "raw_response": r["text"],
                           "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                           "latency_s": r["latency_s"]})
        return r["text"]

    def run_visible(self, code, task):
        self.tool_calls += 1
        ok, _, err, _ = run_code_with_asserts(code, task["test_imports"], task["test_list"][:1], TIMEOUT_S)
        self.steps.append({"step": "tool:run_visible_test", "ok": ok, "error": err})
        return ok, err


# --------------------------------------------------------------------------- memory (static store)
_TOK = re.compile(r"[a-z]+")
_STOP = set("a an the to of in for and or is are be by with from given write function python that this which "
            "its it as on at".split())


def _toks(s: str) -> set:
    return {w for w in _TOK.findall(s.lower()) if w not in _STOP and len(w) > 2}


class MemoryBank:
    """Static store of solved tasks (reference code) disjoint from the online pool and the held-out set.

    Static on purpose: the outcome of (task, pattern) must not depend on the order in which the bandit saw
    tasks, otherwise one cached outcome table could not serve every policy and seed identically.
    """

    def __init__(self, items: list[dict], k: int = 2):
        self.items = items
        self.k = k
        self._t = [_toks(it["prompt"]) for it in items]

    def retrieve(self, prompt: str) -> list[dict]:
        q = _toks(prompt)
        scored = [(len(q & t) / (len(q | t) or 1), -it["task_id"], it) for t, it in zip(self._t, self.items)]
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [it for _, _, it in scored[: self.k]]


# --------------------------------------------------------------------------- patterns
def p_direct(rec, task, mem):
    return extract_code(rec.lm("solve", SOLVER, solve_user_msg(task)))


def p_plan_execute(rec, task, mem):
    plan = rec.lm("plan", PLANNER, solve_user_msg(task))
    return extract_code(rec.lm("execute", SOLVER, f"{solve_user_msg(task)}\n\nFollow this plan:\n{plan.strip()}"))


def p_self_refine(rec, task, mem):
    code = extract_code(rec.lm("solve", SOLVER, solve_user_msg(task)))
    user = f"Task:\n{solve_user_msg(task)}\n\nSolution:\n```python\n{code}\n```"
    review = rec.lm("refine", CRITIC, user)
    # self-refine stopping rule: a review without a code block ("already correct") keeps the current code
    return extract_code(review) if "```" in review else code


def p_verifier_loop(rec, task, mem):
    code = extract_code(rec.lm("solve", SOLVER, solve_user_msg(task)))
    for i in range(MAX_REPAIRS + 1):
        ok, err = rec.run_visible(code, task)
        if ok or i == MAX_REPAIRS:
            return code
        user = (f"Task:\n{solve_user_msg(task)}\n\nPrevious solution:\n```python\n{code}\n```\n\n"
                f"Error:\n{err or 'unknown error'}")
        fix = rec.lm(f"repair{i + 1}", REPAIR, user)
        code = extract_code(fix) if "```" in fix else code
    return code


def p_memory(rec, task, mem):
    rec.tool_calls += 1
    shots = mem.retrieve(task["prompt"])
    rec.steps.append({"step": "tool:memory_retrieve", "retrieved": [s["task_id"] for s in shots]})
    ex = "\n\n".join(f"### Example {i + 1}\nTask: {s['prompt']}\nTest: {s['test_list'][0]}\n"
                     f"```python\n{s['code'].strip()}\n```" for i, s in enumerate(shots))
    return extract_code(rec.lm("solve", MEMORY, f"{ex}\n\n### New task\n{solve_user_msg(task)}"))


FUNCS = {"direct": p_direct, "plan_execute": p_plan_execute, "self_refine": p_self_refine,
         "verifier_loop": p_verifier_loop, "memory": p_memory}


def run_pattern(pattern: str, task: dict, client, mem: MemoryBank) -> dict:
    """Run one pattern on one task, then the shared verifier (all asserts). Never raises on model errors."""
    rec = Recorder(client)
    t0 = time.time()
    code = FUNCS[pattern](rec, task, mem)
    ok, n_passed, err, _ = run_code_with_asserts(code, task["test_imports"], task["test_list"], TIMEOUT_S)
    lm_steps = [s for s in rec.steps if "prompt_tokens" in s]
    return {
        "task_id": task["task_id"], "pattern": pattern,
        "success": int(ok), "n_asserts_passed": n_passed, "n_asserts": len(task["test_list"]),
        "tokens": sum(s["prompt_tokens"] + s["completion_tokens"] for s in lm_steps),
        "prompt_tokens": sum(s["prompt_tokens"] for s in lm_steps),
        "completion_tokens": sum(s["completion_tokens"] for s in lm_steps),
        "lm_calls": len(lm_steps), "tool_calls": rec.tool_calls,
        "latency_s": round(sum(s["latency_s"] for s in lm_steps), 3),
        "wall_s": round(time.time() - t0, 3),
        "final_code": code, "verifier_error": err, "steps": rec.steps,
    }
