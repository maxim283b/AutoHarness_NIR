from types import SimpleNamespace

from harness_adapter import HarnessConfig, MBPPHarnessAdapter, extract_code

TASK = {
    "task_id": 1,
    "prompt": "Write a function to add two numbers.",
    "test_imports": [],
    "test_list": ["assert add(1, 2) == 3", "assert add(5, 5) == 10", "assert add(-1, 1) == 0  # HIDDEN"],
}
CAND = {"solver_prompt": "S", "repair_prompt": "R"}


def fake_lm(responses):
    it = iter(responses)

    def completion(**kw):
        if isinstance(r := next(it), Exception):
            raise r
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=r), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    return completion


def test_extract_code():
    assert extract_code("x\n```python\ndef f(): pass\n```\ny") == "def f(): pass"
    assert extract_code("```\na=1\n```") == "a=1"
    assert extract_code("```python\ntruncated") == "truncated"


def test_pass_without_repair(tmp_path):
    ad = MBPPHarnessAdapter(HarnessConfig(), log_path=tmp_path / "t.jsonl",
                            completion_fn=fake_lm(["```python\ndef add(a, b):\n    return a + b\n```"]))
    eb = ad.evaluate([TASK], CAND, capture_traces=True)
    assert len(eb.outputs) == len(eb.scores) == len(eb.trajectories) == 1
    assert eb.scores == [1.0] and not eb.outputs[0]["repair_used"]
    assert eb.outputs[0]["prompt_tokens"] == 10
    assert len((tmp_path / "t.jsonl").read_text().splitlines()) == 1


def test_repair_used_and_hidden_asserts_not_leaked():
    ad = MBPPHarnessAdapter(HarnessConfig(), completion_fn=fake_lm([
        "```python\ndef add(a, b):\n    return a - b\n```",
        "```python\ndef add(a, b):\n    return abs(a) + b\n```",  # passes visible, fails hidden
    ]))
    eb = ad.evaluate([TASK], CAND, capture_traces=True)
    out = eb.outputs[0]
    assert out["repair_used"] and eb.scores == [0.0] and out["n_lm_calls"] == 2
    assert "Previous solution" in out["steps"][1]["messages"][1]["content"]
    ds = ad.make_reflective_dataset(CAND, eb, ["solver_prompt", "repair_prompt"])
    blob = str(ds) + str([s["messages"] for s in out["steps"]])
    assert "HIDDEN" not in blob and "add(5, 5)" not in blob


def test_no_repair_mode():
    ad = MBPPHarnessAdapter(HarnessConfig(use_repair=False), completion_fn=fake_lm(["```python\ndef add(a,b): return 0\n```"]))
    eb = ad.evaluate([TASK], CAND)
    assert eb.scores == [0.0] and eb.outputs[0]["n_lm_calls"] == 1 and eb.trajectories is None


def test_lm_error_gives_zero_not_raise():
    ad = MBPPHarnessAdapter(HarnessConfig(), completion_fn=fake_lm([RuntimeError("boom"), RuntimeError("boom")]))
    eb = ad.evaluate([TASK, TASK], CAND)
    assert eb.scores == [0.0, 0.0]
    assert "boom" in eb.outputs[0]["harness_error"]


def test_repair_dataset_falls_back_when_no_repair_triggered():
    ad = MBPPHarnessAdapter(HarnessConfig(), completion_fn=fake_lm(["```python\ndef add(a, b):\n    return a + b\n```"]))
    eb = ad.evaluate([TASK], CAND, capture_traces=True)
    ds = ad.make_reflective_dataset(CAND, eb, ["repair_prompt"])
    assert len(ds["repair_prompt"]) == 1 and ds["repair_prompt"][0]["Repair attempt"] == "(not triggered)"
