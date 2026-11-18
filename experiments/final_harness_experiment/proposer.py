from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .models import COMPONENTS, HarnessCandidate


FORBIDDEN_FEEDBACK_KEYS = {"reference_sql", "hidden_tests", "hidden_test", "answer", "held_out_label"}


class Proposer(Protocol):
    calls: int

    def propose(
        self,
        parent: HarnessCandidate,
        component: str,
        safe_feedback: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> HarnessCandidate: ...


def validate_safe_feedback(value: Any) -> None:
    if isinstance(value, dict):
        bad = {str(k).lower() for k in value} & FORBIDDEN_FEEDBACK_KEYS
        if bad:
            raise ValueError(f"forbidden feedback fields: {sorted(bad)}")
        for item in value.values():
            validate_safe_feedback(item)
    elif isinstance(value, list):
        for item in value:
            validate_safe_feedback(item)


@dataclass
class LMStudioProposer:
    endpoint: str = "http://127.0.0.1:1234/v1/chat/completions"
    model: str = "ministral-3-8b-instruct-2512"
    temperature: float = 0.0
    seed: int = 20261012
    timeout_s: float = 180.0
    calls: int = 0

    def propose(
        self,
        parent: HarnessCandidate,
        component: str,
        safe_feedback: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> HarnessCandidate:
        if component not in COMPONENTS:
            raise ValueError(component)
        validate_safe_feedback(safe_feedback)
        compact_history = history[-5:]
        prompt = {
            "task": "Improve exactly one reusable harness component for unseen SQL tasks.",
            "component": component,
            "current_text": parent.components[component],
            "safe_verifier_feedback": safe_feedback,
            "recent_optimizer_history": compact_history,
            "constraints": [
                "Do not include task answers, SQL solutions, hidden fields, or benchmark-specific constants.",
                "Return JSON with keys new_text and reason only.",
                "The new text must be a general instruction, not a solution to an example.",
            ],
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You improve frozen-model agent harness instructions under a strict JSON contract."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "temperature": self.temperature,
            "seed": self.seed + self.calls,
            "max_tokens": 384,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            output = json.loads(response.read())
        self.calls += 1
        text = output["choices"][0]["message"]["content"]
        parsed = json.loads(text)
        if set(parsed) != {"new_text", "reason"}:
            raise ValueError("proposer output must contain exactly new_text and reason")
        return parent.apply({component: str(parsed["new_text"]).strip()}, str(parsed["reason"]).strip())


@dataclass
class DeterministicProposer:
    """Credential-free proposer used only by tests and blocked-gate smoke."""

    calls: int = 0

    def propose(self, parent, component, safe_feedback, history):
        validate_safe_feedback(safe_feedback)
        self.calls += 1
        suffix = f" Generalize the rule; optimizer step {self.calls}."
        return parent.apply({component: parent.components[component] + suffix}, "deterministic smoke mutation")


@dataclass
class SharedGEPAProposer:
    """One proposer contract shared by GEPA, random search and both RL arms."""

    client: LMStudioProposer
    call_log: list[dict[str, Any]]

    @property
    def calls(self) -> int:
        return self.client.calls

    def __call__(self, candidate, reflective_dataset, components_to_update, **_kwargs):
        updates: dict[str, str] = {}
        parent = HarnessCandidate(dict(candidate))
        for component in components_to_update:
            records = [dict(item) for item in reflective_dataset.get(component, [])]
            # Normalize GEPA's rich records into the same safe channel used by
            # the other optimizers. Public task text and generated outputs are
            # allowed; hidden/reference fields are rejected recursively.
            validate_safe_feedback(records)
            child = self.client.propose(parent, component, records, self.call_log)
            updates[component] = child.components[component]
            self.call_log.append({
                "call": self.client.calls,
                "component": component,
                "parent_hash": parent.candidate_hash,
                "child_hash": child.candidate_hash,
                "reason": child.proposer_reason,
            })
            parent = child
        return updates
