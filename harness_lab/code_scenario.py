"""AutoSaddler V2 scenario plugin for the component-map code harness (MBPP)."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence

from harness_lab import _paths  # noqa: F401
from autosaddler.v2.core.domain import (
    ArtifactRef, Case, Cost, Evaluation, JsonValue, Observation, canonical_json, sha256_digest,
)
from autosaddler.v2.core.ports import EvaluationContext, ScenarioComponents
from autosaddler.v2.harness.component_map import ComponentMapHarnessSpace
from autosaddler.v2.prompting.models import SessionSpec
from autosaddler.v2.storage.local import LocalRunStore

from harness_lab.benchmarks import load_humaneval_split, load_mbpp
from harness_lab.code_harness import COMPONENTS, run_task
from harness_lab.llm import LLMSettings, OllamaClient


def _all_tasks() -> dict[str, dict]:
    out = {t["task_id"]: t for split in ("train", "val", "test") for t in load_mbpp(split)}
    out.update({t["task_id"]: t for split in ("train", "val", "test") for t in load_humaneval_split(split)})
    return out


_all_mbpp = _all_tasks


class CodeEvaluator:
    def __init__(self, *, harness_space, store: LocalRunStore, settings: LLMSettings) -> None:
        self.harness_space, self.store, self.settings = harness_space, store, settings
        self.tasks = _all_mbpp()
        self.fingerprint = f"code-harness-ollama/v1:{settings.model}:{settings.temperature}:{settings.seed}"

    async def evaluate(self, candidate, cases: Sequence[Case], context: EvaluationContext) -> Evaluation:
        if not cases or any(case.split != context.split for case in cases):
            raise ValueError("cases must match EvaluationContext.split")
        mat = self.harness_space.materialize(candidate, "evaluate")
        try:
            components = json.loads((mat.root / "candidate.json").read_text(encoding="utf-8"))
        finally:
            mat.release()
        client = OllamaClient(self.settings)
        observations: list[Observation] = []
        for case in cases:
            task = self.tasks[case.case_id]
            for rep in range(context.repetitions):
                done = context.attempt_sink.completed(candidate_id=candidate.candidate_id,
                                                      case_id=case.case_id, repetition=rep)
                if done is not None:
                    observations.append(done)
                    continue
                attempt_id, attempt_no = context.attempt_sink.start(
                    candidate_id=candidate.candidate_id, case_id=case.case_id, repetition=rep)
                try:
                    row = await asyncio.to_thread(run_task, task, components, client)
                    disposition, score = ("success" if row["success"] else "task_failure"), row["reward"]
                except Exception as error:  # infra failure, not a task failure
                    row = {"task_id": case.case_id, "success": None, "reward": None, "tokens": 0,
                           "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0,
                           "infrastructure_error": f"{type(error).__name__}: {error}"}
                    disposition, score = "execution_error", None
                row_id = sha256_digest(canonical_json({"op": context.operation_id, "cand": candidate.candidate_id,
                                                       "case": case.case_id, "rep": rep})).removeprefix("sha256:")
                ref = self.store.write_json(f"raw_target_evaluations/{row_id}.json",
                                            {**row, "split": case.split, "candidate_id": candidate.candidate_id},
                                            kind="raw-target-evaluation")
                cost = Cost(rollouts=1, input_tokens=row["prompt_tokens"], output_tokens=row["completion_tokens"],
                            wall_seconds=row["latency_s"])
                obs = Observation.create(
                    candidate_id=candidate.candidate_id, case_id=case.case_id, split=case.split, repetition=rep,
                    disposition=disposition, score=score, evaluator_fingerprint=self.fingerprint,
                    objectives={"success": float(bool(row["success"]))} if score is not None else {},
                    output=ref, attempts=attempt_no, cost=cost,
                    metadata={"target_tokens": row["tokens"], "repairs": row.get("repairs"),
                              "first_error": (row.get("first_error") or row.get("final_error") or "")[:300],
                              "infrastructure_error": row.get("infrastructure_error")},
                )
                context.attempt_sink.complete(attempt_id, obs, cost)
                observations.append(obs)
        evaluation_id = sha256_digest(context.operation_id)
        prefix = {"train": "evaluations", "development": "quarantine/dev", "test": "post_optimization/test"}[context.split]
        return Evaluation(
            evaluation_id=evaluation_id, candidate_id=candidate.candidate_id, split=context.split,
            purpose=context.purpose, iteration=context.iteration,
            requested_case_ids=tuple(c.case_id for c in cases), observations=tuple(observations),
            artifact_dir=ArtifactRef(uri=f"{prefix}/{evaluation_id.removeprefix('sha256:')}", kind="evaluation-directory"),
        )


class CodeEvidenceBuilder:
    """Train-only evidence. Uses only the visible test of each task (no hidden asserts)."""

    def __init__(self, store: LocalRunStore, tasks: Mapping[str, dict], *, max_chars: int, failures_only: bool) -> None:
        self.store, self.tasks, self.max_chars, self.failures_only = store, tasks, max_chars, failures_only

    def build(self, evaluation: Evaluation) -> ArtifactRef:
        if evaluation.split != "train":
            raise ValueError("Optimization evidence may be built from training only")
        entries, used = [], 0
        for obs in sorted(evaluation.observations, key=lambda o: (o.score is None or o.score > 0, o.case_id)):
            if self.failures_only and obs.score and obs.score > 0:
                continue
            raw = self.store.read_json(obs.output.uri) if obs.output else {}
            task = self.tasks[obs.case_id]
            entry = {
                "case_id": obs.case_id, "passed_all_tests": bool(obs.score and obs.score > 0),
                "task": task["prompt"][:600], "visible_test": task["visible_test"] or "(none; only pass/fail + error class are shown)",
                "first_attempt_code": (raw.get("first_code") or "")[:700],
                "visible_error": (raw.get("first_error") or raw.get("final_error") or "")[:300], "repairs_used": raw.get("repairs"),
            }
            size = len(json.dumps(entry))
            if used + size > self.max_chars:
                break
            used += size
            entries.append(entry)
        scores = [o.score for o in evaluation.observations if o.score is not None]
        payload = {"schema_version": "code-training-evidence/v1", "evaluation_id": evaluation.evaluation_id,
                   "train_pass_rate": sum(scores) / len(scores) if scores else None, "n_cases": len(scores),
                   "cases": entries, "data_boundary": "visible test + its error only; hidden asserts never shown"}
        eid = sha256_digest(canonical_json(payload)).removeprefix("sha256:")
        return self.store.write_json(f"evidence/{eid}/evidence.json", payload, kind="safe-training-evidence")


class CodePromptPack:
    def __init__(self, *, store: LocalRunStore, patch_scope: str) -> None:
        self.store, self.patch_scope = store, patch_scope

    def session(self, kind: str, context: Mapping[str, JsonValue]) -> SessionSpec:
        common = ("You optimize a Python code-generation harness for a small local LLM using TRAINING evidence "
                  "only. Never infer hidden tests or reference solutions.")
        files: dict[str, str] = {"session_context.json": canonical_json(context) + "\n"}
        if kind == "evolve":
            prompt = ("Select exactly one allowed candidate ID from session_context.json. Return "
                      'schema_version "autosaddler-evolution/v1", parent_ids (list with that one ID), '
                      "component_sources as {}, and rationale.")
            schema = _obj(("schema_version", "parent_ids", "component_sources", "rationale"))
        elif kind == "diagnose_patch":
            ev = context.get("evidence")
            if not isinstance(ev, Mapping) or not isinstance(ev.get("uri"), str):
                raise ValueError("diagnose_patch requires an evidence artifact")
            files["safe_training_evidence.json"] = canonical_json(self.store.read_json(str(ev["uri"]))) + "\n"
            scope = ("Change EXACTLY ONE component (best-attributed fix)." if self.patch_scope == "single_component"
                     else "Change one or more components.")
            prompt = (f"Read candidate.json and safe_training_evidence.json. Diagnose a general weakness of the "
                      f"prompts, then rewrite existing text components ({', '.join(COMPONENTS)}). {scope} Keep "
                      'edits short and general. Return {"schema_version": "autosaddler-diagnosis-patch/v1", '
                      '"updates": {<component>: <new full text>}, "diagnosis": "<why>"}.')
            updates = {"type": "object", "minProperties": 1,
                       "propertyNames": {"enum": list(COMPONENTS)},
                       "additionalProperties": {"type": "string", "minLength": 1}}
            if self.patch_scope == "single_component":
                updates["maxProperties"] = 1
            schema = {"type": "object", "required": ["schema_version", "updates", "diagnosis"],
                      "properties": {"schema_version": {"type": "string"}, "updates": updates,
                                     "diagnosis": {"type": "string"}}, "additionalProperties": False}
        elif kind == "reflect":
            prompt = ('Reflect on the aggregate train/development fields in session_context.json. Return '
                      '{"schema_version": "autosaddler-reflection/v1", "lessons": [{"scope": ..., "statement": ..., '
                      '"evidence_case_ids": [<train case ids only>]}]}.')
            schema = _obj(("schema_version", "lessons"))
        else:
            raise ValueError(f"unknown session kind: {kind}")
        return SessionSpec(
            kind=kind, system_context=common, task_prompt=prompt,
            skills={"code-harness-optimization": (
                "---\nname: code-harness-optimization\n"
                'description: "Optimize public code-harness prompts from safe training feedback."\n---\n\n'
                "Preserve the component schema. Use no hidden answers.\n")},
            output_schema=schema, workspace_files=files,
            capabilities=frozenset({"read_workspace", "edit_workspace", "load_skills"}),
            mutation_label="code-harness-component-patch" if kind == "diagnose_patch" else None,
        )


def _obj(required: tuple[str, ...]) -> Mapping[str, JsonValue]:
    return {"type": "object", "required": list(required),
            "properties": {n: {} for n in required}, "additionalProperties": False}


def build_code_components(*, settings, base_dir, run_dir, store, ledger) -> ScenarioComponents:
    del base_dir, ledger
    allowed = {"baseline", "train_case_ids", "development_case_ids", "target", "evidence", "patch_scope"}
    if set(settings) != allowed:
        raise ValueError(f"code_harness settings keys must be {sorted(allowed)}")
    baseline = {str(k): str(v) for k, v in settings["baseline"].items()}
    if set(baseline) != set(COMPONENTS):
        raise ValueError(f"baseline must define exactly {COMPONENTS}")
    t = settings["target"]
    llm = LLMSettings(model=str(t["model"]), base_url=str(t["base_url"]), temperature=float(t["temperature"]),
                      seed=int(t["seed"]), num_ctx=int(t["num_ctx"]), max_tokens=int(t["max_tokens"]),
                      api_key_env=str(t.get("api_key_env", "")))
    tasks = _all_mbpp()
    train_ids = tuple(str(i) for i in settings["train_case_ids"])
    dev_ids = tuple(str(i) for i in settings["development_case_ids"])
    if set(train_ids + dev_ids) - tasks.keys():
        raise ValueError("unknown MBPP task ids")
    harness = ComponentMapHarnessSpace(
        baseline=baseline, store_root=run_dir / "candidates", materialization_root=run_dir / "materialized",
        validator=lambda v: None)
    ev = settings["evidence"]
    return ScenarioComponents(
        name="code_harness", version="1", harness_space=harness,
        evaluator=CodeEvaluator(harness_space=harness, store=store, settings=llm),
        evidence_builder=CodeEvidenceBuilder(store, tasks, max_chars=int(ev["max_chars"]),
                                             failures_only=bool(ev["failures_only"])),
        prompt_pack=CodePromptPack(store=store, patch_scope=str(settings["patch_scope"])),
        train_cases=tuple(Case(case_id=i, split="train", payload={"bench": "mbpp"}) for i in train_ids),
        development_cases=tuple(Case(case_id=i, split="development", payload={"bench": "mbpp"}) for i in dev_ids),
        required_capabilities=frozenset({"read_workspace", "edit_workspace", "load_skills"}),
        resolved_entities={
            "resolved/sources/harness.json": {"type": "structured", "content_digest": sha256_digest(canonical_json(baseline))},
            "resolved/sources/dataset.json": {"type": "mbpp-sanitized-frozen-split", "train": list(train_ids),
                                              "development": list(dev_ids), "test": {"state": "closed"}},
            "resolved/target_runtime.json": {"client": "OllamaClient", "model": llm.model, "temperature": llm.temperature,
                                             "seed": llm.seed, "num_ctx": llm.num_ctx},
            "resolved/prompts/evolve.md": "# Evolve\n\nSelect an accepted parent from training state only.\n",
            "resolved/prompts/diagnosis_patch.md": "# Diagnose and patch\n\nPatch prompt components from safe evidence.\n",
            "resolved/prompts/reflect.md": "# Reflect\n\nRecord lessons from aggregate evidence.\n",
        },
    )
