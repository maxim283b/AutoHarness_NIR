from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gepa.core.adapter import EvaluationBatch

from experiments.harness_optimization.engine import LMStudioClient, ModelSettings, evaluate_sql_task
from experiments.harness_optimization.genome import HarnessGenome

from .models import COMPONENTS, digest


@dataclass(frozen=True)
class ComponentMapHarnessGenome(HarnessGenome):
    """A real HarnessGenome whose system prompt is the public component map.

    The legacy core restricts ``system_prompt_template`` to four preset prompts.
    AutoSaddler must be able to change public text, so this adapter preserves all
    runtime invariants while giving the composed text an explicit template ID.
    """

    def validate(self) -> None:
        if self.system_prompt_template != "autosaddler_component_map/v1":
            raise ValueError("unexpected component-map template")
        if not self.system_prompt.strip():
            raise ValueError("component-map prompt must be non-empty")
        if self.pattern != "verifier_loop" or self.max_repairs != 1:
            raise ValueError("Gate A2 fixes the real runner to one verifier repair")
        if self.planning_mode != "none" or self.memory_k != 0:
            raise ValueError("Gate A2 planning and memory settings drifted")
        if not self.use_visible_verifier or self.verifier_trigger != "after_each_attempt":
            raise ValueError("Gate A2 verifier settings drifted")
        if self.stop_rule != "max_repairs_or_success":
            raise ValueError("Gate A2 stop rule drifted")
        if self.tool_policy != {
            "mode": "read_only_sql_single_statement",
            "allowed_prefixes": ["SELECT", "WITH"],
            "allow_mutation": False,
        }:
            raise ValueError("Gate A2 tool policy drifted")
        if self.context_budget != 2048 or self.generation_max_tokens != 384:
            raise ValueError("Gate A2 token budgets drifted")


def genome_from_candidate(candidate: dict[str, str], context_budget: int = 2048) -> HarnessGenome:
    missing = set(COMPONENTS) - candidate.keys()
    if missing:
        raise ValueError(f"missing harness components: {sorted(missing)}")
    system = "\n\n".join(candidate[name] for name in COMPONENTS)
    return ComponentMapHarnessGenome(
        system_prompt=system,
        system_prompt_template="autosaddler_component_map/v1",
        pattern="verifier_loop",
        planning_mode="none",
        max_repairs=1,
        memory_k=0,
        use_visible_verifier=True,
        verifier_trigger="after_each_attempt",
        stop_rule="max_repairs_or_success",
        tool_policy={
            "mode": "read_only_sql_single_statement",
            "allowed_prefixes": ["SELECT", "WITH"],
            "allow_mutation": False,
        },
        context_budget=context_budget,
        generation_max_tokens=384,
    )


class SQLGEPAAdapter:
    """Official GEPA adapter with deterministic SQL verification and cached target calls."""

    propose_new_texts = None

    def __init__(self, *, model_seed: int, log_path: Path, context_budget: int = 2048):
        self.settings = ModelSettings(seed=model_seed, context_budget=context_budget)
        self.client = LMStudioClient(self.settings)
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict[str, Any]] = {}
        self.target_calls = 0
        self.target_tokens = 0

    def _evaluate_one(self, task: dict[str, Any], candidate: dict[str, str]) -> dict[str, Any]:
        key = digest({"candidate": candidate, "task_id": task["task_id"], "seed": self.settings.seed})
        if key in self.cache:
            return self.cache[key]
        genome = genome_from_candidate(candidate, self.settings.context_budget)
        row = evaluate_sql_task(task, genome, self.client, self.settings)
        row.update({"cache_key": key, "candidate_hash": digest(candidate), "model_seed": self.settings.seed})
        self.cache[key] = row
        self.target_calls += int(row["target_calls"])
        self.target_tokens += int(row["tokens"])
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return row

    def evaluate(self, batch, candidate, capture_traces=False):
        rows = [self._evaluate_one(task, candidate) for task in batch]
        trajectories = None
        if capture_traces:
            trajectories = [
                {
                    "task_id": row["task_id"],
                    "public_task": row["public_task"],
                    "generated_output": row["final_output"],
                    "success": row["success"],
                    "error_class": row.get("error_code"),
                    "tokens": row["tokens"],
                    "tool_calls": row["tool_calls"],
                    "reward": row["reward"],
                }
                for row in rows
            ]
        return EvaluationBatch(
            outputs=[{"task_id": row["task_id"], "success": row["success"]} for row in rows],
            scores=[float(row["reward"]) for row in rows],
            trajectories=trajectories,
            objective_scores=[{"success": float(row["success"]), "reward": float(row["reward"])} for row in rows],
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = [
            {
                "public_task": trace["public_task"],
                "generated_output": trace["generated_output"],
                "feedback": {
                    "success": trace["success"],
                    "error_class": trace["error_class"],
                    "tokens": trace["tokens"],
                    "tool_calls": trace["tool_calls"],
                    "reward": trace["reward"],
                },
            }
            for trace in (eval_batch.trajectories or [])
        ]
        return {component: records for component in components_to_update}
