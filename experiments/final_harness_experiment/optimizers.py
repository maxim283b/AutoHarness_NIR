from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .models import COMPONENTS, HarnessCandidate
from .proposer import Proposer


@dataclass(frozen=True, slots=True)
class Transition:
    state: str
    action: str
    parent_hash: str
    child_hash: str
    reward: float
    accepted: bool
    next_state: str


def mean_reward(rows: list[dict[str, Any]]) -> float:
    return sum(float(row["reward"]) for row in rows) / len(rows)


def tail_aware_reward(rows: list[dict[str, Any]]) -> float:
    rewards = sorted(float(row["reward"]) for row in rows)
    count = max(1, math.ceil(0.20 * len(rewards)))
    tail = sum(rewards[:count]) / count
    return 0.8 * (sum(rewards) / len(rewards)) + 0.2 * tail


@dataclass
class SequentialRLOptimizer:
    seed: int
    learning_rate: float = 0.20
    baseline_rate: float = 0.20
    temperature: float = 0.70
    objective: Callable[[list[dict[str, Any]]], float] = mean_reward
    theta: dict[str, dict[str, float]] = field(default_factory=dict)
    baselines: dict[str, float] = field(default_factory=dict)
    transitions: list[Transition] = field(default_factory=list)

    def _state(self, candidate: HarnessCandidate, score: float, step: int) -> str:
        score_bin = math.floor(score * 20) / 20
        recent = ",".join(t.action for t in self.transitions[-3:]) or "none"
        return f"h={candidate.candidate_hash[:12]}|score={score_bin:.2f}|step={min(step, 14)}|recent={recent}"

    def _distribution(self, state: str) -> list[float]:
        preferences = self.theta.setdefault(state, {action: 0.0 for action in COMPONENTS})
        logits = [preferences[action] / self.temperature for action in COMPONENTS]
        top = max(logits)
        weights = [math.exp(value - top) for value in logits]
        total = sum(weights)
        return [weight / total for weight in weights]

    def choose_action(self, state: str, rng: random.Random) -> tuple[str, list[float]]:
        probabilities = self._distribution(state)
        point = rng.random()
        cumulative = 0.0
        for action, probability in zip(COMPONENTS, probabilities, strict=True):
            cumulative += probability
            if point <= cumulative:
                return action, probabilities
        return COMPONENTS[-1], probabilities

    def _update(self, state: str, action: str, probabilities: list[float], reward: float) -> None:
        baseline = self.baselines.get(state, 0.0)
        advantage = reward - baseline
        self.baselines[state] = baseline + self.baseline_rate * advantage
        preferences = self.theta[state]
        for candidate_action, probability in zip(COMPONENTS, probabilities, strict=True):
            gradient = (1.0 if candidate_action == action else 0.0) - probability
            preferences[candidate_action] += self.learning_rate * advantage * gradient

    def run(
        self,
        seed_candidate: HarnessCandidate,
        proposer: Proposer,
        evaluate: Callable[[HarnessCandidate], list[dict[str, Any]]],
        steps: int,
    ) -> HarnessCandidate:
        if steps < 4:
            raise ValueError("RL requires at least four dependent steps")
        rng = random.Random(self.seed)
        current = seed_candidate
        current_rows = evaluate(current)
        current_score = self.objective(current_rows)
        history: list[dict[str, Any]] = []
        for step in range(steps):
            state = self._state(current, current_score, step)
            action, probabilities = self.choose_action(state, rng)
            safe_feedback = [
                {
                    "task_id": row.get("task_id"),
                    "success": bool(row.get("success")),
                    "error_class": row.get("error_class"),
                    "tokens": int(row.get("tokens", 0)),
                    "tool_calls": int(row.get("tool_calls", 0)),
                }
                for row in current_rows
            ]
            child = proposer.propose(current, action, safe_feedback, history)
            child_rows = evaluate(child)
            child_score = self.objective(child_rows)
            delta = child_score - current_score
            accepted = delta > 0
            next_candidate = child if accepted else current
            next_score = child_score if accepted else current_score
            next_state = self._state(next_candidate, next_score, step + 1)
            self._update(state, action, probabilities, delta)
            transition = Transition(state, action, current.candidate_hash, child.candidate_hash, delta, accepted, next_state)
            self.transitions.append(transition)
            history.append({"action": action, "reward": delta, "accepted": accepted, "child_hash": child.candidate_hash})
            current, current_rows, current_score = next_candidate, (child_rows if accepted else current_rows), next_score
        return current

    def policy_state(self) -> dict[str, Any]:
        return {
            "algorithm": "on_policy_tabular_REINFORCE",
            "theta": self.theta,
            "baselines": self.baselines,
            "transitions": [asdict(t) for t in self.transitions],
        }


class TailAwareRLOptimizer(SequentialRLOptimizer):
    """Tail-aware RL variant; explicitly not published TailRL."""

    def __init__(self, seed: int, **kwargs: Any) -> None:
        super().__init__(seed=seed, objective=tail_aware_reward, **kwargs)

    def policy_state(self) -> dict[str, Any]:
        value = super().policy_state()
        value["algorithm"] = "tail_aware_tabular_REINFORCE_not_TailRL"
        return value
