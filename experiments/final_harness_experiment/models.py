from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


COMPONENTS = (
    "system_instruction",
    "planning_instruction",
    "verification_instruction",
    "repair_rule",
    "tool_rule",
    "stop_rule",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class HarnessCandidate:
    components: dict[str, str]
    parent_hash: str | None = None
    patch: dict[str, str] = field(default_factory=dict)
    proposer_reason: str = "seed"

    def __post_init__(self) -> None:
        missing = set(COMPONENTS) - self.components.keys()
        extra = self.components.keys() - set(COMPONENTS)
        if missing or extra:
            raise ValueError(f"invalid component set; missing={sorted(missing)}, extra={sorted(extra)}")
        if any(not isinstance(v, str) or not v.strip() for v in self.components.values()):
            raise ValueError("all harness components must be non-empty strings")
        if any(len(v) > 8000 for v in self.components.values()):
            raise ValueError("component exceeds the preregistered 8000-character limit")

    @property
    def candidate_hash(self) -> str:
        return digest(self.components)

    def apply(self, updates: dict[str, str], reason: str) -> "HarnessCandidate":
        if not updates or not set(updates).issubset(COMPONENTS):
            raise ValueError("patch must update at least one known component")
        merged = dict(self.components)
        merged.update(updates)
        return HarnessCandidate(merged, self.candidate_hash, dict(updates), reason)

    def public_record(self) -> dict[str, Any]:
        return {
            "candidate_hash": self.candidate_hash,
            "parent_hash": self.parent_hash,
            "components": self.components,
            "patch": self.patch,
            "proposer_reason": self.proposer_reason,
        }


def seed_candidate() -> HarnessCandidate:
    return HarnessCandidate(
        {
            "system_instruction": "Produce one correct read-only SQLite query and no prose.",
            "planning_instruction": "Privately identify tables, joins, filters, grouping, and ordering before answering.",
            "verification_instruction": "Check every identifier against the public schema and check that the result answers the request.",
            "repair_rule": "When given a safe verifier error class, fix the root cause and return a complete replacement query.",
            "tool_rule": "Use only a single read-only SELECT or WITH statement; never modify the database.",
            "stop_rule": "Stop at the first verified success or when the fixed repair budget is exhausted.",
        }
    )
