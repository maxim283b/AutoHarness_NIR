"""End-to-end GEPA run on mocks: callbacks, patches log and token stopper work."""
import json

import gepa

from data import load_splits
from harness_adapter import MBPPHarnessAdapter
from llm import TokenLedger
from mock_lm import MockTarget, MockTeacher
from run_optimize import PatchLogger, TokenBudgetStopper

SEED = {"solver_prompt": "Solve it.", "repair_prompt": "Fix it."}


def run(tmp_path, mode, max_target=None):
    s = load_splits()
    ledger = TokenLedger()

    class T(MockTarget):
        def chat(self, m):
            r = super().chat(m)
            ledger.add("target", r["prompt_tokens"], r["completion_tokens"], 0)
            return r

    ad = MBPPHarnessAdapter(T(), tmp_path / "tr.jsonl", {"run_id": "e2e"}, {t["task_id"] for t in s["train"]}, mode)
    stop = TokenBudgetStopper(ledger, max_target, None)
    res = gepa.optimize(seed_candidate=SEED, trainset=s["train"], valset=s["val"], adapter=ad,
                        reflection_lm=MockTeacher(), max_metric_calls=80, stop_callbacks=[stop],
                        reflection_minibatch_size=3, candidate_selection_strategy="pareto",
                        module_selector="round_robin", run_dir=str(tmp_path / "state"), seed=0,
                        cache_evaluation=True, callbacks=[PatchLogger(tmp_path, ledger)])
    return res, ledger, stop


def test_e2e_standard(tmp_path):
    res, ledger, _ = run(tmp_path, "standard")
    assert res.val_aggregate_scores[res.best_idx] > res.val_aggregate_scores[0]
    patches = [json.loads(l) for l in (tmp_path / "patches.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(p["decision"] == "accepted" for p in patches)
    assert all("diff" in p for p in patches if p["decision"] != "no_decision")


def test_e2e_contrastive_and_budget(tmp_path):
    res, ledger, stop = run(tmp_path, "contrastive", max_target=1500)
    assert stop.reason == "target_token_budget"
    assert res.total_metric_calls < 80
