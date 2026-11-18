"""Persistent optimizer boundary for staged AutoSaddler screening.

The official engine, provider sessions, patch application, acceptance and reflection
remain untouched.  This controller only chooses parent/mutation strategy before the
official sessions and persists an RL update after development evaluation.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

from autosaddler.v2.core.events import RunEvent

from .models import COMPONENTS


METHODS = (
    "original_autosaddler_optimizer",
    "random_search",
    "tabular_rl_optimizer",
    "tail_aware_rl_variant",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class ScreeningController:
    def __init__(self, *, method: str, seed: int, run_dir: Path) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown screening method: {method}")
        self.method = method
        self.seed = seed
        self.run_dir = run_dir
        self.path = run_dir / "optimizer_policy_state.json"
        self.log_path = run_dir / "optimizer_events.jsonl"
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {
            "schema_version": "screening-optimizer-state/v1",
            "method": self.method,
            "seed": self.seed,
            "q_values": {component: 0.0 for component in COMPONENTS},
            "counts": {component: 0 for component in COMPONENTS},
            "baseline": 0.0,
            "history": [],
            "pending": None,
            "last_updated_iteration": -1,
        }

    def _save(self) -> None:
        _atomic_json(self.path, self.state)

    def _append(self, value: dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _score_state(self, iteration: int, current_hash: str) -> dict[str, Any]:
        history = self.state["history"]
        recent = history[-3:]
        return {
            "current_harness_hash": current_hash,
            "safe_score_history": [item["mean_reward"] for item in recent],
            "safe_cost_history": [item["target_tokens"] for item in recent],
            "search_state": {"candidate_index": iteration, "last_actions": [item["action"] for item in recent]},
        }

    def _choose_action(self, iteration: int, current_hash: str) -> tuple[str, dict[str, float], dict[str, Any]]:
        state = self._score_state(iteration, current_hash)
        if self.method == "original_autosaddler_optimizer":
            return "official_unconstrained_patch", {}, state
        if self.method == "random_search":
            action = random.Random(f"{self.seed}:{iteration}").choice(COMPONENTS)
            return action, {item: 1.0 / len(COMPONENTS) for item in COMPONENTS}, state
        q_values = self.state["q_values"]
        temperature = 0.7
        logits = [float(q_values[item]) / temperature for item in COMPONENTS]
        maximum = max(logits)
        weights = [math.exp(item - maximum) for item in logits]
        total = sum(weights)
        probabilities = {item: weight / total for item, weight in zip(COMPONENTS, weights, strict=True)}
        point = random.Random(f"{self.seed}:{iteration}:rl").random()
        cumulative = 0.0
        action = COMPONENTS[-1]
        for name in COMPONENTS:
            cumulative += probabilities[name]
            if point <= cumulative:
                action = name
                break
        return action, probabilities, state

    def prepare(self, *, iteration: int, candidate_ids: list[str]) -> dict[str, Any]:
        pending = self.state.get("pending")
        if isinstance(pending, dict) and pending.get("iteration") == iteration:
            return pending
        parent = candidate_ids[-1]
        action, probabilities, state = self._choose_action(iteration, parent)
        pending = {
            "iteration": iteration,
            "candidate_index": iteration + 1,
            "parent_candidate_id": parent,
            "action": action,
            "action_probabilities": probabilities,
            "state": state,
        }
        self.state["pending"] = pending
        self._save()
        self._append({"event": "ActionSelected", **pending, "policy_state": self.public_state()})
        return pending

    def wrap_prompt_pack(self, base):
        controller = self

        class ControlledPromptPack:
            def session(self, kind, context):
                spec = base.session(kind, context)
                if kind == "evolve":
                    candidate_ids = context.get("candidate_ids")
                    if not isinstance(candidate_ids, list) or not candidate_ids:
                        raise ValueError("evolve candidate list missing")
                    pending = controller.prepare(iteration=int(context["iteration"]), candidate_ids=candidate_ids)
                    if controller.method != "original_autosaddler_optimizer":
                        spec = replace(
                            spec,
                            task_prompt=spec.task_prompt
                            + f"\nThe optimizer selected parent {pending['parent_candidate_id']}; use exactly that parent.",
                        )
                elif kind == "diagnose_patch" and controller.method != "original_autosaddler_optimizer":
                    pending = controller.state.get("pending")
                    if not isinstance(pending, dict):
                        raise RuntimeError("optimizer action missing before diagnose_patch")
                    spec = replace(
                        spec,
                        task_prompt=spec.task_prompt
                        + f"\nThe optimizer selected mutation strategy `{pending['action']}`. Update exactly that component.",
                    )
                return spec

        return ControlledPromptPack()

    def transition_hook(self, event: RunEvent) -> None:
        if event.event_type != "DeferredWorkScheduled":
            return
        iteration = int(event.payload["owning_iteration"])
        if iteration <= int(self.state.get("last_updated_iteration", -1)):
            return
        self._update_from_obligation(iteration, dict(event.payload))

    def reconcile(self, events) -> None:
        for event in events:
            if event.event_type == "DeferredWorkScheduled":
                iteration = int(event.payload["owning_iteration"])
                if iteration > int(self.state.get("last_updated_iteration", -1)):
                    self._update_from_obligation(iteration, dict(event.payload))

    def _development_rows(self, candidate_id: str) -> list[dict[str, Any]]:
        rows = []
        directory = self.run_dir / "raw_target_evaluations"
        if not directory.exists():
            return rows
        for path in directory.glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("candidate_id") == candidate_id and ":development" in str(row.get("operation_id")):
                rows.append(row)
        return rows

    def _candidate_metrics(self, candidate_id: str) -> tuple[float, float, int]:
        rows = [row for row in self._development_rows(candidate_id) if row.get("reward") is not None]
        if not rows:
            return 0.0, 0.0, 0
        rewards = sorted(float(row["reward"]) for row in rows)
        mean = sum(rewards) / len(rewards)
        count = max(1, math.ceil(0.20 * len(rewards)))
        tail = sum(rewards[:count]) / count
        tokens = sum(int(row.get("tokens", 0)) for row in rows)
        return mean, tail, tokens

    def _update_from_obligation(self, iteration: int, payload: dict[str, Any]) -> None:
        pending = self.state.get("pending")
        if not isinstance(pending, dict) or int(pending.get("iteration", -1)) != iteration:
            raise RuntimeError(f"missing persisted action for iteration {iteration}")
        child_id = str(payload["candidate_id"])
        parent_id = str(payload["working_parent_candidate_id"])
        child_mean, child_tail, target_tokens = self._candidate_metrics(child_id)
        parent_mean, parent_tail, _ = self._candidate_metrics(parent_id)
        mean_delta = child_mean - parent_mean
        tail_delta = child_tail - parent_tail
        if self.method == "tail_aware_rl_variant":
            reward = 0.8 * mean_delta + 0.2 * tail_delta
        else:
            reward = mean_delta
        action = str(pending["action"])
        if self.method in {"tabular_rl_optimizer", "tail_aware_rl_variant"}:
            baseline = float(self.state["baseline"])
            advantage = reward - baseline
            self.state["baseline"] = baseline + 0.2 * advantage
            probabilities = pending["action_probabilities"]
            for name in COMPONENTS:
                gradient = (1.0 if name == action else 0.0) - float(probabilities[name])
                self.state["q_values"][name] = float(self.state["q_values"][name]) + 0.2 * advantage * gradient
            self.state["counts"][action] = int(self.state["counts"][action]) + 1
        record = {
            "candidate_index": iteration + 1,
            "iteration": iteration,
            "parent_candidate_id": parent_id,
            "child_candidate_id": child_id,
            "action": action,
            "accepted": bool(payload["accepted_by_minibatch_gate"]),
            "mean_reward": child_mean,
            "tail_reward": child_tail,
            "mean_delta": mean_delta,
            "tail_delta": tail_delta,
            "policy_reward": reward,
            "target_tokens": target_tokens,
            "proposed_patch": payload.get("updates"),
        }
        self.state["history"].append(record)
        self.state["last_updated_iteration"] = iteration
        self.state["pending"] = None
        self._save()
        self._append({"event": "PolicyUpdated", **record, "policy_state": self.public_state()})

    def public_state(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "q_values": self.state["q_values"],
            "counts": self.state["counts"],
            "baseline": self.state["baseline"],
            "last_updated_iteration": self.state["last_updated_iteration"],
            "pending": self.state["pending"],
        }
