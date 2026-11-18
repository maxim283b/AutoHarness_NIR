from harness_lab.code_harness import SEED_COMPONENTS, extract_code, run_task
from harness_lab.stats import mcnemar_exact, paired_compare, wilson


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, **kw):
        text = self.replies.pop(0)
        rec = {"text": text, "prompt_tokens": 10, "completion_tokens": 5, "latency_s": 0.0}
        self.calls.append(rec)
        return rec


TASK = {"task_id": "t", "prompt": "add", "imports": [], "tests": ["assert add(1,2)==3", "assert add(2,2)==4"],
        "visible_test": "assert add(1,2)==3"}


def test_extract_code():
    assert extract_code("x\n```python\nprint(1)\n```\ny") == "print(1)"


def test_repair_loop_fixes_wrong_first_attempt():
    bad = "```python\ndef add(a,b):\n    return a-b\n```"
    good = "```python\ndef add(a,b):\n    return a+b\n```"
    row = run_task(TASK, SEED_COMPONENTS, FakeClient([bad, good]))
    assert row["success"] and row["repairs"] == 1 and row["target_calls"] == 2


def test_no_repair_keeps_failure():
    bad = "```python\ndef add(a,b):\n    return a-b\n```"
    row = run_task(TASK, SEED_COMPONENTS, FakeClient([bad]), use_repair=False)
    assert not row["success"] and row["repairs"] == 0


def test_hidden_test_failure_is_detected():
    sneaky = "```python\ndef add(a,b):\n    return 3\n```"  # passes visible, fails hidden
    row = run_task(TASK, SEED_COMPONENTS, FakeClient([sneaky]))
    assert not row["success"] and row["first_pass_ok"]


def test_stats():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 5) < 0.07
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi
    r = paired_compare([0, 0, 1, 1], [1, 1, 1, 1], boots=200)
    assert r["delta"] == 0.5 and r["b_only"] == 2
