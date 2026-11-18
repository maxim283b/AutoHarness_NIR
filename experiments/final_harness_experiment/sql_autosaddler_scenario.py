"""Real SQL scenario adapter for the official AutoSaddler V2 engine.

This module does not propose or repair patches.  It translates the public
component map to the project's HarnessGenome, executes the existing LM Studio
runner and SQL verifier, and exposes only safe aggregate training evidence to
the unchanged official AutoSaddler session loop.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from autosaddler.v2.core.domain import (
    ArtifactRef,
    Case,
    Cost,
    Evaluation,
    JsonValue,
    Observation,
    canonical_json,
    sha256_digest,
)
from autosaddler.v2.core.ports import EvaluationContext, ScenarioComponents
from autosaddler.v2.harness.component_map import ComponentMapHarnessSpace
from autosaddler.v2.prompting.models import SessionSpec
from autosaddler.v2.storage.local import LocalRunStore

from experiments.final_harness_experiment.models import COMPONENTS
from experiments.final_harness_experiment.sql_gepa_adapter import genome_from_candidate
from experiments.harness_optimization.engine import LMStudioClient, ModelSettings, evaluate_sql_task
from experiments.harness_optimization.genome import content_hash
from pilot_sql.tasks import load_tasks


class RecordingLMStudioClient(LMStudioClient):
    """The real project client plus an append-only in-memory copy of raw replies."""

    is_real_target_client = True

    def __init__(self, settings: ModelSettings) -> None:
        super().__init__(settings)
        self.raw_calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
        response = super().chat(messages, max_tokens)
        self.raw_calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "raw_model_output": response["text"],
                "prompt_tokens": response["prompt_tokens"],
                "completion_tokens": response["completion_tokens"],
                "latency_s": response["latency_s"],
                "finish_reason": response["finish_reason"],
            }
        )
        return response


class SQLHarnessEvaluator:
    def __init__(
        self,
        *,
        harness_space: ComponentMapHarnessSpace,
        store: LocalRunStore,
        settings: ModelSettings,
        target_client_factory=RecordingLMStudioClient,
    ) -> None:
        self.harness_space = harness_space
        self.store = store
        self.settings = settings
        self.target_client_factory = target_client_factory
        self.tasks = {task["task_id"]: task for task in load_tasks()}
        self.fingerprint = f"sql-harness-lmstudio/v1:{settings.config_hash}"

    async def evaluate(
        self,
        candidate,
        cases: Sequence[Case],
        context: EvaluationContext,
    ) -> Evaluation:
        if not cases or any(case.split != context.split for case in cases):
            raise ValueError("SQL evaluator cases must match EvaluationContext.split")
        materialized = self.harness_space.materialize(candidate, "evaluate")
        try:
            components = json.loads((materialized.root / "candidate.json").read_text(encoding="utf-8"))
        finally:
            materialized.release()
        genome = genome_from_candidate(components, self.settings.context_budget)
        genome_config = asdict(genome)
        genome_hash = content_hash(genome_config)
        client = self.target_client_factory(self.settings)
        if not getattr(client, "is_real_target_client", False):
            raise RuntimeError("Gate A2 forbids a mock target client")

        observations: list[Observation] = []
        for case in cases:
            task = self.tasks[case.case_id]
            for repetition in range(context.repetitions):
                cached = context.attempt_sink.completed(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    repetition=repetition,
                )
                if cached is not None:
                    observations.append(cached)
                    continue
                attempt_id, attempt_number = context.attempt_sink.start(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    repetition=repetition,
                )
                call_start = len(client.raw_calls)
                try:
                    row = evaluate_sql_task(task, genome, client, self.settings)
                    row["infrastructure_error"] = None
                    disposition = "success" if row["success"] else "task_failure"
                    score: float | None = float(row["reward"])
                except Exception as error:  # infrastructure failures are not task failures
                    row = {
                        "task_id": case.case_id,
                        "success": None,
                        "reward": None,
                        "tokens": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "target_calls": 0,
                        "tool_calls": 0,
                        "final_output": None,
                        "verifier_events": [],
                        "model_messages": [],
                        "infrastructure_error": {"type": type(error).__name__, "message": str(error)},
                    }
                    disposition = "execution_error"
                    score = None
                raw_calls = client.raw_calls[call_start:]
                public_row = {
                    **row,
                    "schema_version": "gate-a2-sql-evaluation-row/v1",
                    "run_id": context.run_id,
                    "operation_id": context.operation_id,
                    "candidate_id": candidate.candidate_id,
                    "candidate_artifact_sha256": candidate.artifact.sha256,
                    "harness_config": genome_config,
                    "harness_config_hash": genome_hash,
                    "target_client": type(client).__name__,
                    "target_model": self.settings.model_id,
                    "target_endpoint": self.settings.endpoint,
                    "split": case.split,
                    "repetition": repetition,
                    "raw_target_calls": raw_calls,
                    "reference_sql_exposed": False,
                }
                row_id = sha256_digest(
                    canonical_json(
                        {
                            "operation": context.operation_id,
                            "candidate": candidate.candidate_id,
                            "case": case.case_id,
                            "repetition": repetition,
                        }
                    )
                ).removeprefix("sha256:")
                row_ref = self.store.write_json(
                    f"raw_target_evaluations/{row_id}.json",
                    public_row,
                    kind="raw-target-evaluation",
                )
                cost = Cost(
                    rollouts=1,
                    input_tokens=int(row.get("prompt_tokens") or 0),
                    output_tokens=int(row.get("completion_tokens") or 0),
                    wall_seconds=float(row.get("latency_s") or 0.0),
                )
                observation = Observation.create(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    split=case.split,
                    repetition=repetition,
                    disposition=disposition,
                    score=score,
                    evaluator_fingerprint=self.fingerprint,
                    objectives=(
                        {"success": float(bool(row["success"])), "reward": float(row["reward"])}
                        if score is not None
                        else {}
                    ),
                    output=row_ref,
                    attempts=attempt_number,
                    cost=cost,
                    metadata={
                        "evaluator": self.fingerprint,
                        "error_code": row.get("error_code"),
                        "target_calls": int(row.get("target_calls") or 0),
                        "target_tokens": int(row.get("tokens") or 0),
                        "tool_calls": int(row.get("tool_calls") or 0),
                        "harness_config_hash": genome_hash,
                        "infrastructure_error": row.get("infrastructure_error"),
                    },
                )
                context.attempt_sink.complete(attempt_id, observation, cost)
                observations.append(observation)

        evaluation_id = sha256_digest(context.operation_id)
        prefix = {"train": "evaluations", "development": "quarantine/dev", "test": "post_optimization/test"}[
            context.split
        ]
        return Evaluation(
            evaluation_id=evaluation_id,
            candidate_id=candidate.candidate_id,
            split=context.split,
            purpose=context.purpose,
            iteration=context.iteration,
            requested_case_ids=tuple(case.case_id for case in cases),
            observations=tuple(observations),
            artifact_dir=ArtifactRef(uri=f"{prefix}/{evaluation_id.removeprefix('sha256:')}", kind="evaluation-directory"),
        )


class SafeSQLEvidenceBuilder:
    def __init__(self, store: LocalRunStore) -> None:
        self.store = store

    def build(self, evaluation: Evaluation) -> ArtifactRef:
        if evaluation.split != "train":
            raise ValueError("Optimization evidence may be built from training only")
        payload = {
            "schema_version": "safe-sql-training-evidence/v1",
            "evaluation_id": evaluation.evaluation_id,
            "observations": [
                {
                    "case_id": item.case_id,
                    "disposition": item.disposition,
                    "score": item.score,
                    "error_code": item.metadata.get("error_code"),
                    "target_calls": item.metadata.get("target_calls"),
                    "target_tokens": item.metadata.get("target_tokens"),
                    "tool_calls": item.metadata.get("tool_calls"),
                }
                for item in evaluation.observations
            ],
            "data_boundary": "safe aggregate verifier feedback only; no gold queries, labels, or non-search data",
        }
        evidence_id = sha256_digest(canonical_json(payload)).removeprefix("sha256:")
        return self.store.write_json(f"evidence/{evidence_id}/evidence.json", payload, kind="safe-training-evidence")


class SQLHarnessPromptPack:
    def __init__(self, *, store: LocalRunStore) -> None:
        self.store = store

    def session(self, kind: str, context: Mapping[str, JsonValue]) -> SessionSpec:
        common = (
            "You optimize a public SQL agent harness using training evidence only. "
            "Never seek or infer gold queries, hidden answers, validation labels, or non-search data. "
            "Write the required JSON object to .autosaddler/session_output.json and nothing else is accepted."
        )
        files: dict[str, str] = {"session_context.json": canonical_json(context) + "\n"}
        if kind == "evolve":
            prompt = (
                "Select exactly one allowed candidate ID from session_context.json. "
                "Return schema_version autosaddler-evolution/v1, parent_ids, empty component_sources, and rationale."
            )
            schema = _object_schema(("schema_version", "parent_ids", "component_sources", "rationale"))
        elif kind == "diagnose_patch":
            evidence = context.get("evidence")
            if not isinstance(evidence, Mapping) or not isinstance(evidence.get("uri"), str):
                raise ValueError("diagnose_patch requires an evidence artifact")
            evidence_payload = self.store.read_json(str(evidence["uri"]))
            files["safe_training_evidence.json"] = canonical_json(evidence_payload) + "\n"
            prompt = (
                "Read candidate.json and safe_training_evidence.json. Diagnose a general harness weakness, then make a "
                "meaningful change to one or more existing text components. Do not add keys. Return schema_version "
                "autosaddler-diagnosis-patch/v1, non-empty updates, and diagnosis. The patch must be useful without "
                "knowing any correct SQL answer."
            )
            schema = {
                "type": "object",
                "required": ["schema_version", "updates", "diagnosis"],
                "properties": {
                    "schema_version": {"type": "string"},
                    "updates": {
                        "type": "object",
                        "minProperties": 1,
                        "propertyNames": {"enum": list(COMPONENTS)},
                        "additionalProperties": {"type": "string", "minLength": 1},
                    },
                    "diagnosis": {"type": "string"},
                },
                "additionalProperties": False,
            }
        elif kind == "reflect":
            prompt = (
                "Reflect only on the aggregate train/development fields in session_context.json. Return "
                "schema_version autosaddler-reflection/v1 and lessons. Every lesson must contain scope, statement, "
                "and evidence_case_ids drawn only from train_case_ids."
            )
            schema = _object_schema(("schema_version", "lessons"))
        else:
            raise ValueError(f"unknown session kind: {kind}")
        return SessionSpec(
            kind=kind,
            system_context=common,
            task_prompt=prompt,
            skills={
                "sql-harness-optimization": (
                    "---\nname: sql-harness-optimization\n"
                    'description: "Optimize public SQL harness instructions from safe verifier feedback."\n---\n\n'
                    "Preserve the component schema. Use no hidden answers. Emit the requested structured contract.\n"
                )
            },
            output_schema=schema,
            workspace_files=files,
            capabilities=frozenset({"read_workspace", "edit_workspace", "load_skills"}),
            mutation_label="sql-harness-component-patch" if kind == "diagnose_patch" else None,
        )


def build_sql_harness_components(*, settings, base_dir, run_dir, store, ledger) -> ScenarioComponents:
    del base_dir, ledger
    required = {"baseline", "train_case_ids", "development_case_ids", "target"}
    if set(settings) != required:
        raise ValueError(f"SQL scenario settings keys must be {sorted(required)}")
    baseline = settings["baseline"]
    if not isinstance(baseline, Mapping) or set(baseline) != set(COMPONENTS):
        raise ValueError("SQL baseline must contain the frozen six-component harness schema")
    target = settings["target"]
    if not isinstance(target, Mapping):
        raise TypeError("target settings must be a mapping")
    model_settings = ModelSettings(
        model_id=str(target["model_id"]),
        endpoint=str(target["endpoint"]),
        temperature=float(target["temperature"]),
        seed=int(target["seed"]),
        context_budget=int(target["context_budget"]),
        generation_max_tokens=int(target["generation_max_tokens"]),
        timeout_s=float(target.get("timeout_s", 180.0)),
    )
    harness = ComponentMapHarnessSpace(
        baseline={str(key): str(value) for key, value in baseline.items()},
        store_root=run_dir / "candidates",
        materialization_root=run_dir / "materialized",
        validator=lambda value: genome_from_candidate(dict(value), model_settings.context_budget).validate(),
    )
    task_ids = {task["task_id"] for task in load_tasks()}
    train_ids = tuple(str(item) for item in settings["train_case_ids"])
    dev_ids = tuple(str(item) for item in settings["development_case_ids"])
    if set(train_ids + dev_ids) - task_ids:
        raise ValueError("scenario contains unknown SQL task IDs")
    return ScenarioComponents(
        name="sql_harness",
        version="1",
        harness_space=harness,
        evaluator=SQLHarnessEvaluator(harness_space=harness, store=store, settings=model_settings),
        evidence_builder=SafeSQLEvidenceBuilder(store),
        prompt_pack=SQLHarnessPromptPack(store=store),
        train_cases=tuple(Case(case_id=item, split="train", payload={"source": "fixed-search-manifest"}) for item in train_ids),
        development_cases=tuple(Case(case_id=item, split="development", payload={"source": "fixed-search-manifest"}) for item in dev_ids),
        required_capabilities=frozenset({"read_workspace", "edit_workspace", "load_skills"}),
        resolved_entities={
            "resolved/sources/harness.json": {
                "type": "structured",
                "space": "component_map_to_HarnessGenome",
                "content_digest": sha256_digest(canonical_json(baseline)),
            },
            "resolved/sources/dataset.json": {
                "type": "pilot_sql_real_verifier",
                "search_manifest": "pilot_sql/splits_manifest.json",
                "train_case_ids": list(train_ids),
                "development_case_ids": list(dev_ids),
                "validation": {"state": "closed"},
                "held_out": {"state": "opaque", "opened": False},
            },
            "resolved/target_runtime.json": {
                "client": "LMStudioClient",
                "model_id": model_settings.model_id,
                "endpoint": model_settings.endpoint,
                "settings_hash": model_settings.config_hash,
            },
            "resolved/prompts/evolve.md": "# Evolve\n\nSelect an accepted parent using training state only.\n",
            "resolved/prompts/diagnosis_patch.md": "# Diagnose and patch\n\nPatch public harness components using safe verifier feedback only.\n",
            "resolved/prompts/reflect.md": "# Reflect\n\nRecord lessons from aggregate training evidence only.\n",
        },
    )


def _object_schema(required: tuple[str, ...]) -> Mapping[str, JsonValue]:
    return {
        "type": "object",
        "required": list(required),
        "properties": {name: {} for name in required},
        "additionalProperties": False,
    }
