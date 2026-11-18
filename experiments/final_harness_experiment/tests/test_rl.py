from __future__ import annotations

from experiments.final_harness_experiment.models import COMPONENTS, seed_candidate
from experiments.final_harness_experiment.optimizers import SequentialRLOptimizer, TailAwareRLOptimizer
from experiments.final_harness_experiment.proposer import DeterministicProposer


def evaluator(candidate):
    # Deterministic fake metric for policy/lineage tests only. More explicit
    # verification text monotonically improves the fake reward.
    bonus = candidate.components["verification_instruction"].count("optimizer step") * 0.05
    return [
        {"task_id": f"s{i}", "success": bonus > 0, "reward": bonus - i * 0.01,
         "tokens": 10, "tool_calls": 1, "error_class": None if bonus else "wrong_result"}
        for i in range(5)
    ]


def test_rl_has_dependent_transitions_and_updates_policy():
    optimizer = SequentialRLOptimizer(seed=7, temperature=0.5)
    final = optimizer.run(seed_candidate(), DeterministicProposer(), evaluator, steps=6)
    state = optimizer.policy_state()
    assert len(optimizer.transitions) == 6
    assert state["algorithm"] == "on_policy_tabular_REINFORCE"
    assert all(t.action in COMPONENTS for t in optimizer.transitions)
    assert any(any(abs(v) > 0 for v in actions.values()) for actions in optimizer.theta.values())
    assert final.candidate_hash


def test_rl_rejects_bandit_length_horizon():
    optimizer = SequentialRLOptimizer(seed=0)
    try:
        optimizer.run(seed_candidate(), DeterministicProposer(), evaluator, steps=3)
    except ValueError as exc:
        assert "four dependent steps" in str(exc)
    else:
        raise AssertionError("short non-sequential horizon was accepted")


def test_tail_variant_is_not_mislabeled_tailrl():
    optimizer = TailAwareRLOptimizer(seed=3)
    optimizer.run(seed_candidate(), DeterministicProposer(), evaluator, steps=4)
    assert optimizer.policy_state()["algorithm"] == "tail_aware_tabular_REINFORCE_not_TailRL"
