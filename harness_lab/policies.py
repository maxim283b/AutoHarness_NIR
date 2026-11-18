"""Low-resource acceptance policies for AutoSaddler.

With 5-12 training tasks per mini-batch a single flipped task moves the mean by
8-20 pp, so the stock `matched_valid_strict_improvement` rule (after > before)
accepts noise.  `NetWinsAcceptance` demands a minimum *net* number of per-task wins.
"""
from __future__ import annotations

from harness_lab import _paths  # noqa: F401
from autosaddler.v2.core.domain import Evaluation, PatchVerdict


class NetWinsAcceptance:
    def __init__(self, min_net_wins: int) -> None:
        self.min_net_wins = min_net_wins

    def compare(self, parent: Evaluation, child: Evaluation) -> PatchVerdict:
        if parent.split != "train" or child.split != "train":
            raise ValueError("Acceptance may compare training evaluations only")
        if parent.requested_case_ids != child.requested_case_ids:
            raise ValueError("Parent and child must be evaluated on the identical ordered case set")
        p = {(o.case_id, o.repetition): o for o in parent.observations}
        c = {(o.case_id, o.repetition): o for o in child.observations}
        keys = [k for k in sorted(p) if k in c and p[k].is_valid and c[k].is_valid]
        if not keys:
            raise ValueError("Acceptance has no matched valid observations")
        wins = sum(1 for k in keys if c[k].score > p[k].score)
        losses = sum(1 for k in keys if c[k].score < p[k].score)
        before = sum(p[k].score for k in keys) / len(keys)
        after = sum(c[k].score for k in keys) / len(keys)
        accepted = (wins - losses) >= self.min_net_wins and after > before
        return PatchVerdict(
            before_score=before, after_score=after,
            compared_case_ids=tuple(dict.fromkeys(k[0] for k in keys)),
            accepted=accepted,
            reason=f"net wins {wins - losses} (wins={wins}, losses={losses}), need >= {self.min_net_wins}",
        )


def register(registry) -> None:
    for k in (1, 2, 3):
        registry.acceptance[f"net_wins_{k}"] = (lambda k=k: NetWinsAcceptance(k))


class FailureFirstSelection:
    """Low-resource batch sampler: fill each mini-batch with training cases the *seed* fails.

    With a ~75%-accurate target a random 4-task batch often has no failure, so no diagnosis
    session can run and the iteration is wasted.  The seed is scored once on the train pool
    (cached, counted as `precompute_rollouts`) and failing cases are rotated through batches.
    """

    def __init__(self, *, batch_size: int, seed: int) -> None:
        self.batch_size, self.seed, self._order = batch_size, seed, None
        self.precompute_rollouts = 0

    def _prepare(self, cases) -> None:
        from harness_lab.code_scenario import _all_mbpp
        from harness_lab.code_harness import SEED_COMPONENTS
        from harness_lab.evaluate import evaluate
        from harness_lab.llm import LLMSettings
        from harness_lab._paths import RESULTS
        tasks = _all_mbpp()
        he = cases[0].case_id.startswith("humaneval")
        rows = evaluate(dict(SEED_COMPONENTS), [tasks[c.case_id] for c in cases], LLMSettings(),
                        use_repair=not he, cache=RESULTS / "cache" / ("humaneval.jsonl" if he else "mbpp.jsonl"))
        self.precompute_rollouts = len(rows)
        fail = [c.case_id for c, r in zip(cases, rows) if not r["success"]]
        ok = [c.case_id for c, r in zip(cases, rows) if r["success"]]
        self._order = (tuple(fail), tuple(ok))

    def select(self, cases, iteration: int):
        from autosaddler.v2.core.policies import TaskSelection
        if self._order is None:
            self._prepare(cases)
        fail, ok = self._order
        n_fail = min(len(fail), max(1, (self.batch_size + 1) // 2 + 1))
        picked = [fail[(iteration * n_fail + i) % len(fail)] for i in range(n_fail)] if fail else []
        picked = list(dict.fromkeys(picked))
        rest = [c for c in ok + fail if c not in picked]
        k = len(rest)
        start = (iteration * self.batch_size) % k if k else 0
        for i in range(k):
            if len(picked) >= min(self.batch_size, len(cases)):
                break
            picked.append(rest[(start + i) % k])
        return TaskSelection(case_ids=tuple(picked), provenance={
            "policy": "failure_first", "iteration": iteration, "seed_failures": len(fail),
            "precompute_rollouts": self.precompute_rollouts})


def register_selection(registry) -> None:
    registry.task_selection["failure_first"] = lambda *, batch_size, seed: FailureFirstSelection(
        batch_size=batch_size, seed=seed)
