import json

import pytest

from data import load_splits
from harness_adapter import MBPPHarnessAdapter
from mock_lm import MockTarget

SEED = {"solver_prompt": "Solve it.", "repair_prompt": "Fix it."}


@pytest.fixture
def splits():
    return load_splits()


@pytest.mark.parametrize("mode", ["standard", "contrastive"])
def test_lengths_and_no_hidden_asserts(splits, tmp_path, mode):
    train = splits["train"][:6]
    ad = MBPPHarnessAdapter(MockTarget(), tmp_path / "t.jsonl", {"run_id": "x"}, {t["task_id"] for t in train}, mode)
    ad.evaluate(train, {"solver_prompt": "CAREFUL", "repair_prompt": "Fix"}, capture_traces=True)  # fills archive
    eb = ad.evaluate(train, SEED, capture_traces=True)
    assert len(eb.outputs) == len(eb.scores) == len(eb.trajectories) == 6
    assert 0 < sum(eb.scores) < 6
    for comp in ("solver_prompt", "repair_prompt"):
        ds = ad.make_reflective_dataset(SEED, eb, [comp])
        assert list(ds) == [comp] and ds[comp]
        text = json.dumps(ds)
        for t in train:
            for hidden in t["test_list"][1:]:
                assert hidden not in text
    lines = (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    for k in ("run_id", "task_id", "candidate_hash", "step", "messages", "raw_response", "extracted_code",
              "visible_test_ok", "final_pass", "prompt_tokens", "completion_tokens"):
        assert k in rec


def test_contrastive_pairs_with_archive(splits, tmp_path):
    train = splits["train"][:6]
    ad = MBPPHarnessAdapter(MockTarget(), None, {}, {t["task_id"] for t in train}, "contrastive")
    ad.evaluate(train, {"solver_prompt": "CAREFUL", "repair_prompt": "Fix"}, capture_traces=True)
    eb = ad.evaluate(train, SEED, capture_traces=True)
    ds = ad.make_reflective_dataset(SEED, eb, ["solver_prompt"])["solver_prompt"]
    paired = [r for r in ds if "Contrasting successful attempt" in r]
    assert paired and all(r["Outcome"] == "FAILED" for r in paired)
    assert "SAME task" in paired[0]["Contrasting successful attempt"]["Source"]
    assert ds[0]["Outcome"] == "FAILED"


def test_standard_has_no_contrast(splits, tmp_path):
    train = splits["train"][:6]
    ad = MBPPHarnessAdapter(MockTarget(), None, {}, {t["task_id"] for t in train}, "standard")
    ad.evaluate(train, {"solver_prompt": "CAREFUL", "repair_prompt": "Fix"}, capture_traces=True)
    eb = ad.evaluate(train, SEED, capture_traces=True)
    ds = ad.make_reflective_dataset(SEED, eb, ["solver_prompt"])["solver_prompt"]
    assert all("Contrasting successful attempt" not in r and "Outcome" not in r for r in ds)


def test_exception_gives_zero(splits):
    class Boom:
        def chat(self, m):
            raise ValueError("bad")
    ad = MBPPHarnessAdapter(Boom(), None, {}, set(), "standard")
    eb = ad.evaluate(splits["train"][:2], SEED, capture_traces=True)
    assert eb.scores == [0.0, 0.0]
