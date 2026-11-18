"""H2 memory must alter later optimizer requests without changing H1."""
import json
from pathlib import PurePosixPath

import pytest

from autosaddler.v2.core.adaptive_memory import AdaptiveReflectionMemory
from autosaddler.v2.core.events import RunEvent
from experiments.autosaddler_reproduction.opencode_prompt_pack import NativeOpenCodePromptPack


class Events:
    def __init__(self):
        self.run_id = "improved-test"
        self.items = []

    def events(self):
        return tuple(self.items)

    def events_of_type(self, kind):
        return tuple(e for e in self.items if e.event_type == kind)

    def append(self, kind, op, payload):
        existing = [e for e in self.items if e.event_type == kind and e.operation_id == op]
        if existing:
            assert existing[0].payload == payload
            return existing[0]
        event = RunEvent.create(sequence=len(self.items)+1, run_id=self.run_id,
                                event_type=kind, operation_id=op, payload=payload)
        self.items.append(event)
        return event


def seed_iteration(store):
    store.append("DeferredWorkScheduled", "scheduled-0", {
        "obligation_id": "reflect-0", "owning_iteration": 0, "candidate_id": "child",
        "working_parent_candidate_id": "parent", "train_case_ids": ["train:a", "train:b"],
        "diagnosis": "grep tool failed on the active path", "updates": {"grep": "fallback"},
        "changed_components": ["opencode-src/packages/core/src/tool/grep.ts"],
        "train_before_aggregate": 0.0, "train_after_aggregate": 0.5,
        "train_before_case_scores": {"train:a": 0.0, "train:b": 1.0},
        "train_after_case_scores": {"train:a": 1.0, "train:b": 0.0},
        "parent_development_aggregate": 0.25, "candidate_development_aggregate": 0.25,
    })
    store.append("IterationCompleted", "completed-0", {
        "iteration": 0, "outcome": "declined", "accepted": False,
    })


def test_journal_reconcile_and_stage_retrieval_are_idempotent():
    store = Events()
    seed_iteration(store)
    memory = AdaptiveReflectionMemory(store)
    memory.reconcile()
    memory.reconcile()
    assert len([e for e in store.items if e.payload.get("kind") == "iteration"]) == 1
    diagnosis = memory.retrieve("diagnose_patch", {"iteration": 1, "train_case_ids": ["train:a"]})
    record = diagnosis["records"][0]
    assert record["fixed_case_ids"] == ["train:a"]
    assert record["regressed_case_ids"] == ["train:b"]
    assert record["causal_status"] == "uncertain"
    assert "grep tool failed" in next(iter(memory.selected_details(diagnosis).values()))
    with pytest.raises(ValueError):
        memory.record_reflection(0, {"status": "confirmed", "behavioral_mechanism": "",
                                     "evidence": "", "evidence_case_ids": []}, ("train:a",))
    assessment = {"status": "uncertain", "behavioral_mechanism": "fallback path was not isolated",
                  "evidence": "same dev score; two changed train outcomes", "evidence_kind": "insufficient",
                  "evidence_case_ids": ["train:a", "train:b"], "source_refs": ["trace-a"],
                  "next_step": "Check active grep call route"}
    memory.record_reflection(0, assessment, ("train:a", "train:b"))
    assert memory.retrieve("patch", {"iteration": 1, "train_case_ids": ["train:a"]})["records"][0]["next_step"] == assessment["next_step"]


def test_h2_prompt_contains_retrieved_evidence_and_causal_schema():
    store = Events()
    seed_iteration(store)
    kwargs = {"store": store, "writable_paths": (PurePosixPath("opencode-src"),),
              "capability_phase_iterations": 1}
    h1 = NativeOpenCodePromptPack(**kwargs)
    h2 = NativeOpenCodePromptPack(**kwargs, adaptive_reflection_memory=True, optimizer_model="GLM-5.3")
    context = {"iteration": 1, "train_case_ids": ["train:a"], "candidate_ids": ["parent"]}
    h1_spec = h1.session("evolve", context)
    h2_spec = h2.session("evolve", context)
    assert ".autosaddler/adaptive_memory/evolve.json" not in h1_spec.workspace_files
    assert "optimizer is Kimi" in h1_spec.system_context
    assert "optimizer is GLM-5.3" in h2_spec.system_context
    retrieval = json.loads(h2_spec.workspace_files[".autosaddler/adaptive_memory/evolve.json"])
    assert retrieval["records"][0]["source_operation_id"]
    detail = retrieval["records"][0]["detail_path"]
    assert detail in h2_spec.workspace_files
    reflect = h2.session("reflect", {"iteration": 0, "train_case_ids": ["train:a"],
                                    "candidate_ids": ["child"]})
    assert "causal_assessment" in reflect.output_schema["required"]
    assert ".autosaddler/adaptive_memory/reflect.json" in reflect.workspace_files
