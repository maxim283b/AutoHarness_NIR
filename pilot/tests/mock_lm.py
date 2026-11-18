"""Deterministic mocks: target solves tasks only if the solver prompt contains 'CAREFUL' (or via repair)."""
from data import load_splits

GOLD = {t["task_id"]: t["code"] for s in load_splits().values() for t in s}


class MockTarget:
    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        sys_p, user = messages[0]["content"], messages[1]["content"]
        tid = next(k for k, v in TASKS.items() if v in user)
        good = ("CAREFUL" in sys_p) or (tid % 3 == 0) or ("Previous solution" in user and tid % 2 == 0)
        code = GOLD[tid] if good else "def broken():\n    return None"
        return {"text": f"```python\n{code}\n```", "prompt_tokens": 50, "completion_tokens": 20,
                "latency_s": 0.0, "done_reason": "stop"}


TASKS = {t["task_id"]: t["prompt"] for s in load_splits().values() for t in s}


class MockTeacher:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return "```\nBe CAREFUL. Write a correct function and return one ```python``` block.\n```"
