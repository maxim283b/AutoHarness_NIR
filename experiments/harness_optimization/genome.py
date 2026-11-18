"""Typed, content-addressed harness candidates and deterministic mutations."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from dataclasses import dataclass, field, replace
from typing import Any


PROMPTS = {
    "seed_sql": (
        "You are a careful SQLite analyst. Return exactly one read-only SELECT or WITH query, "
        "with no explanation and no markdown fence."
    ),
    "schema_grounded": (
        "You are a precise SQLite analyst. Ground every table and column in the supplied schema. "
        "Return exactly one read-only SELECT or WITH query and nothing else."
    ),
    "constraint_first": (
        "Write one safe read-only SQLite query. Before answering, privately check filters, grouping, "
        "ordering, aliases, and boundary conditions. Output only SELECT or WITH SQL."
    ),
    "verify_columns": (
        "Solve the SQLite task exactly. Privately verify selected columns, join keys, aggregation, "
        "and requested ordering against the schema. Return one read-only query only."
    ),
}

PATTERNS = ("direct", "plan_execute", "self_refine", "verifier_loop")
PLANNING_MODES = ("none", "concise_private_plan", "checklist")
VERIFIER_TRIGGERS = ("final_only", "after_each_attempt")
STOP_RULES = ("first_success", "max_repairs_or_success")
CONTEXT_BUDGETS = (2048, 4096)
GENERATION_MAX_TOKENS = (256, 384)
MUTATION_OPERATORS = (
    "prompt_template",
    "pattern",
    "planning_mode",
    "repair_budget",
    "visible_verifier",
    "verifier_trigger",
    "stop_rule",
    "generation_budget",
)


def canonical_json(value: Any) -> str:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HarnessGenome:
    system_prompt: str
    system_prompt_template: str
    pattern: str
    planning_mode: str
    max_repairs: int
    memory_k: int
    use_visible_verifier: bool
    verifier_trigger: str
    stop_rule: str
    tool_policy: dict[str, Any] = field(default_factory=lambda: {
        "mode": "read_only_sql_single_statement",
        "allowed_prefixes": ["SELECT", "WITH"],
        "allow_mutation": False,
    })
    context_budget: int = 2048
    generation_max_tokens: int = 384

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def candidate_hash(self) -> str:
        return content_hash(self.to_dict())

    def validate(self) -> None:
        if self.system_prompt_template not in PROMPTS:
            raise ValueError("unknown system_prompt_template")
        if self.system_prompt != PROMPTS[self.system_prompt_template]:
            raise ValueError("system_prompt does not match its frozen template")
        if self.pattern not in PATTERNS:
            raise ValueError("unsupported pattern")
        if self.planning_mode not in PLANNING_MODES:
            raise ValueError("unsupported planning_mode")
        if self.max_repairs not in (0, 1, 2):
            raise ValueError("max_repairs must be 0..2")
        if self.pattern != "verifier_loop" and self.max_repairs != 0:
            raise ValueError("repairs are available only in verifier_loop")
        if self.memory_k != 0:
            raise ValueError("memory is disabled in the preregistered core to avoid cross-candidate leakage")
        if self.verifier_trigger not in VERIFIER_TRIGGERS:
            raise ValueError("unsupported verifier_trigger")
        if self.stop_rule not in STOP_RULES:
            raise ValueError("unsupported stop_rule")
        if self.tool_policy != {
            "mode": "read_only_sql_single_statement",
            "allowed_prefixes": ["SELECT", "WITH"],
            "allow_mutation": False,
        }:
            raise ValueError("tool policy may describe allowed calls but may not change tool code")
        if self.context_budget not in CONTEXT_BUDGETS:
            raise ValueError("unsupported context budget")
        if self.generation_max_tokens not in GENERATION_MAX_TOKENS:
            raise ValueError("unsupported generation token budget")


def seed_genome(context_budget: int = 2048) -> HarnessGenome:
    genome = HarnessGenome(
        system_prompt=PROMPTS["seed_sql"],
        system_prompt_template="seed_sql",
        pattern="direct",
        planning_mode="none",
        max_repairs=0,
        memory_k=0,
        use_visible_verifier=False,
        verifier_trigger="final_only",
        stop_rule="first_success",
        context_budget=context_budget,
        generation_max_tokens=384,
    )
    genome.validate()
    return genome


def _next(values: tuple, current: Any, rng: random.Random) -> Any:
    choices = [v for v in values if v != current]
    return choices[rng.randrange(len(choices))]


def mutate_genome(parent: HarnessGenome, operator: str, seed: int) -> HarnessGenome:
    """Apply one deterministic, schema-valid mutation.

    Determinism is defined by `(parent hash, operator, seed)`, so resume never
    depends on process-global RNG state.
    """
    parent.validate()
    if operator not in MUTATION_OPERATORS:
        raise ValueError(f"unknown mutation operator: {operator}")
    rng_seed = int(parent.candidate_hash[:16], 16) ^ int(seed)
    rng = random.Random(rng_seed)
    child = parent
    if operator == "prompt_template":
        name = _next(tuple(PROMPTS), parent.system_prompt_template, rng)
        child = replace(parent, system_prompt_template=name, system_prompt=PROMPTS[name])
    elif operator == "pattern":
        pattern = _next(PATTERNS, parent.pattern, rng)
        child = replace(
            parent,
            pattern=pattern,
            max_repairs=1 if pattern == "verifier_loop" else 0,
            verifier_trigger="after_each_attempt" if pattern == "verifier_loop" else "final_only",
            use_visible_verifier=pattern == "verifier_loop",
        )
    elif operator == "planning_mode":
        child = replace(parent, planning_mode=_next(PLANNING_MODES, parent.planning_mode, rng))
    elif operator == "repair_budget":
        child = replace(
            parent,
            pattern="verifier_loop",
            max_repairs=_next((0, 1, 2), parent.max_repairs, rng),
            verifier_trigger="after_each_attempt",
            use_visible_verifier=True,
            stop_rule="max_repairs_or_success",
        )
    elif operator == "visible_verifier":
        child = replace(parent, use_visible_verifier=not parent.use_visible_verifier)
    elif operator == "verifier_trigger":
        trigger = _next(VERIFIER_TRIGGERS, parent.verifier_trigger, rng)
        if trigger == "after_each_attempt":
            child = replace(parent, pattern="verifier_loop", verifier_trigger=trigger,
                            max_repairs=max(1, parent.max_repairs), use_visible_verifier=True)
        else:
            child = replace(parent, verifier_trigger=trigger)
    elif operator == "stop_rule":
        child = replace(parent, stop_rule=_next(STOP_RULES, parent.stop_rule, rng))
    elif operator == "generation_budget":
        child = replace(parent, generation_max_tokens=_next(GENERATION_MAX_TOKENS, parent.generation_max_tokens, rng))
    child.validate()
    if child.candidate_hash == parent.candidate_hash:
        raise AssertionError("mutation produced an identical child")
    return child

