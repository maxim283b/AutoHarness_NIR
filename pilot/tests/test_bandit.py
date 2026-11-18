import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bandit"))

import simulate  # noqa: E402
from patterns import PATTERNS, MemoryBank, run_pattern  # noqa: E402
from policies import REQUIRED, Thompson  # noqa: E402


def _fake_cache(d: Path, n_online=60, n_heldout=20):
    rng = np.random.default_rng(0)
    p_succ = {"direct": 0.5, "plan_execute": 0.55, "self_refine": 0.5, "verifier_loop": 0.7, "memory": 0.6}
    online, heldout = list(range(1, n_online + 1)), list(range(1001, 1001 + n_heldout))
    types = {i: ["string", "math", "sequence"][i % 3] for i in online + heldout}
    with open(d / "outcomes.jsonl", "w") as f:
        for i in online + heldout:
            for p in PATTERNS:
                f.write(json.dumps({"task_id": i, "pattern": p, "success": int(rng.random() < p_succ[p]),
                                    "tokens": int(rng.integers(100, 900)), "tool_calls": int(p in ("verifier_loop", "memory")),
                                    "lm_calls": 1, "latency_s": float(rng.random() * 5), "harness_hash": "h"}) + "\n")
    (d / "config.json").write_text(json.dumps({
        "harness_hash": "h", "pools": {"online": online, "heldout": heldout},
        "task_types": {str(k): v for k, v in types.items()}, "target_model": "fake", "target_options": {}}))
    return online, heldout


def _run(tmp_path, monkeypatch, tag):
    monkeypatch.setattr(simulate, "HERE", tmp_path)
    return simulate.main(["--seeds", "4", "--budgets", "10,25", "--cache", str(tmp_path), "--tag", tag])


def test_reproducible_and_no_heldout_updates(tmp_path, monkeypatch):
    online, heldout = _fake_cache(tmp_path)
    s1 = _run(tmp_path, monkeypatch, "a")
    s2 = _run(tmp_path, monkeypatch, "b")
    assert s1["aggregate"] == s2["aggregate"]
    log = [json.loads(x) for x in gzip.open(tmp_path / "results" / "a" / "online_log.jsonl.gz", "rt")]
    assert {r["task_id"] for r in log} <= set(online)
    # equal budget: every (seed, budget, policy) has exactly T observations, and all policies see the same stream
    by = {}
    for r in log:
        by.setdefault((r["seed"], r["budget"], r["policy"]), []).append(r["task_id"])
    for (seed, T, pol), ids in by.items():
        assert len(ids) == T
        assert ids == by[(seed, T, "random")]


def test_frozen_policy_rejects_updates():
    p = Thompson(PATTERNS, np.random.default_rng(0), r_lo=-0.5, r_hi=1.0)
    p.update(0, 1.0)
    p.frozen = True
    with pytest.raises(AssertionError):
        p.update(0, 1.0)


@pytest.mark.parametrize("P", REQUIRED)
def test_policies_learn_clear_winner(P):
    rng = np.random.default_rng(1)
    pol = P(PATTERNS, rng, r_lo=0.0, r_hi=1.0)
    for t in range(200):
        a = pol.select(t)
        pol.update(a, float(rng.random() < (0.9 if a == 3 else 0.2)))
    if P.name != "random":
        assert pol.deploy() == 3


class _EchoLM:
    """Returns a fixed reply; lets us check pattern control flow without a real model."""

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, msgs):
        return {"text": self.replies.pop(0), "prompt_tokens": 10, "completion_tokens": 5, "latency_s": 0.0}


TASK = {"task_id": 1, "prompt": "Write a function to add two numbers.", "test_imports": [],
        "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]}
GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
BAD = "```python\ndef add(a, b):\n    return a - b\n```"


def test_self_refine_keeps_code_when_review_has_no_block():
    r = run_pattern("self_refine", TASK, _EchoLM([GOOD, "The solution is already correct."]), MemoryBank([]))
    assert r["success"] == 1 and r["lm_calls"] == 2 and r["tool_calls"] == 0


def test_verifier_loop_counts_tool_calls_and_repairs():
    r = run_pattern("verifier_loop", TASK, _EchoLM([BAD, GOOD]), MemoryBank([]))
    assert r["success"] == 1 and r["lm_calls"] == 2 and r["tool_calls"] == 2
