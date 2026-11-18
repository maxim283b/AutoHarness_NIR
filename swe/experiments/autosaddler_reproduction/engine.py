"""Official AutoSaddler V2 engine with a shared Git-backed OpenCode target.

V2 is explicitly recorded; this is not the paper's V1 binary reproduction.
The smoke mode uses synthetic scores and must never be reported as benchmark data.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile

from autosaddler.v2.config.registry import default_registry, ScenarioRegistration, build_runtime
from autosaddler.v2.core.domain import ArtifactRef, Case, Cost, Evaluation, Observation, sha256_digest
from autosaddler.v2.core.policies import TaskSelection
from autosaddler.v2.core.ports import ScenarioComponents
from autosaddler.v2.harness.git import GitHarnessSpace, GitVerificationVerdict
from autosaddler.v2.plugins.api import ScenarioPlugin
from autosaddler.v2.plugins.meta_are.prompt_pack import MetaAREPromptPack
from autosaddler.v2.plugins.meta_are.evidence import MetaAREEvidenceBuilder
from autosaddler.v2.prompting.models import SessionResult, Usage

from .accounting import UsageLedger
from .benchmarks import BENCHMARKS, selected_benchmarks

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).parent
CAPABILITIES = frozenset({"read_workspace", "edit_workspace", "run_commands", "load_skills"})
WRITABLE = tuple(PurePosixPath(p) for p in ("instructions.md", "opencode.json", ".opencode", "opencode-src"))


class ShortGitHarnessSpace(GitHarnessSpace):
    """MinGit still limits linked-worktree $GIT_DIR paths despite longpaths."""
    def __init__(self, *, short_root, **kwargs):
        self.short_root = short_root
        super().__init__(**kwargs)
    def begin_mutation(self, parent, context):
        context = replace(context, workspace_root=self.short_root / "m" / str(context.iteration))
        return super().begin_mutation(parent, context)
    def capture_attempt_delta(self, session, attempt_workspace, *, attempt_operation_id):
        if len(str(attempt_workspace.resolve())) < 230:
            return super().capture_attempt_delta(session,attempt_workspace,attempt_operation_id=attempt_operation_id)
        # MinGit rejects a long --work-tree root before core.longpaths applies.
        # Copy the completed attempt to an owned short path for read-only diff
        # capture; preserve the original provider trace and published delta.
        parent=(self.short_root/'a').resolve();parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='delta-',dir=parent) as directory:
            short=Path(directory).resolve()
            if not short.is_relative_to(parent): raise RuntimeError('Short delta path escaped its owned root')
            shutil.copytree(attempt_workspace,short,dirs_exist_ok=True)
            return super().capture_attempt_delta(session,short,attempt_operation_id=attempt_operation_id)


def prepare_seed():
    source = ROOT.parent / "tbv7/experiments/autosaddler_reproduction/harness_seed"
    digest = hashlib.sha256(b"".join(p.relative_to(source).as_posix().encode()+p.read_bytes() for p in sorted(source.rglob("*")) if p.is_file())).hexdigest()[:12]
    destination = ROOT / "third_party" / ("experiment_harness_" + digest)
    if not destination.exists():
        shutil.copytree(source, destination)
        subprocess.run(["git", "init", str(destination)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(destination), "config", "core.autocrlf", "false"], check=True)
        subprocess.run(["git", "-C", str(destination), "config", "core.eol", "lf"], check=True)
        subprocess.run(["git", "-C", str(destination), "add", "-f", "."], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(destination), "-c", "user.name=Experiment Setup", "-c", "user.email=experiment@localhost", "commit", "-m", "chore: freeze shared OpenCode baseline"], check=True, capture_output=True)
    if subprocess.check_output(["git", "-C", str(destination), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Seed harness repository is dirty")
    subprocess.run(["git", "-C", str(destination), "config", "core.longpaths", "true"], check=True)
    if subprocess.check_output(["git", "-C", str(destination), "config", "core.autocrlf"], text=True).strip() != 'false':
        raise RuntimeError('Independent source repository must preserve exact canonical bytes')
    return destination, subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()


class BalancedSelection:
    def __init__(self, *, batch_size, seed, benchmarks=BENCHMARKS):
        self.benchmarks = selected_benchmarks({"benchmarks": benchmarks})
        if batch_size < len(self.benchmarks) or batch_size % len(self.benchmarks):
            raise ValueError("Batch size must be a positive multiple of selected benchmark count")
        self.batch_size, self.seed = batch_size, seed
    def select(self, cases, iteration):
        groups = defaultdict(list)
        for case in cases:
            groups[case.payload["benchmark"]].append(case.case_id)
        if set(groups) != set(self.benchmarks):
            raise ValueError("Train sets must match selected benchmarks")
        selected = []
        for bench in self.benchmarks:
            ids = sorted(groups[bench], key=lambda c: hashlib.sha256(f"{self.seed}:{c}".encode()).hexdigest())
            n = min(self.batch_size // len(self.benchmarks), len(ids))
            selected.extend(ids[(iteration*n+i) % len(ids)] for i in range(n))
        return TaskSelection(tuple(selected), {"policy": "balanced_benchmarks", "seed": self.seed, "iteration": iteration})


class MacroRanking:
    def __init__(self, benchmarks=BENCHMARKS):
        self.benchmarks = selected_benchmarks({"benchmarks": benchmarks})

    def select(self, candidates):
        def score(pair):
            evaluation = pair[1]
            groups = defaultdict(list)
            for item in evaluation.observations:
                if not item.is_valid:
                    raise ValueError("Cannot rank candidates with invalid dev attempts")
                groups[item.case_id.split(":", 1)[0]].append(item.score)
            if set(groups) != set(self.benchmarks):
                raise ValueError("Incomplete benchmark development evaluation")
            return sum(sum(v) / len(v) for v in groups.values()) / len(self.benchmarks)
        return max(enumerate(candidates), key=lambda pair: (score(pair[1]), -pair[0]))[1]


class OpenCodePromptPack(MetaAREPromptPack):
    def session(self, kind, context):
        spec = super().session(kind, context)
        context_text = "\nThe optimization target is the shared OpenCode harness, not Meta-ARE. Benchmark adapters, model IDs, evaluators and split manifests are immutable. Apply capability patches through custom OpenCode tools/skills, steering patches through instructions.md. Never create benchmark-specific winning configurations.\n"
        return replace(spec, system_context=spec.system_context + context_text)


class JointEvaluator:
    def __init__(self, space, store, settings):
        self.space, self.store, self.settings = space, store, settings
        self.fingerprint = sha256_digest(json.dumps(settings, sort_keys=True))
    async def evaluate(self, candidate, cases, context):
        if any(case.split != context.split for case in cases):
            raise ValueError("Case split mismatch")
        observations = []
        materialized = self.space.materialize(candidate, "evaluate")
        try:
            for case in cases:
                for repetition in range(context.repetitions):
                    cached = context.attempt_sink.completed(candidate_id=candidate.candidate_id, case_id=case.case_id, repetition=repetition)
                    if cached is not None:
                        observations.append(cached)
                        continue
                    attempt_id, number = context.attempt_sink.start(candidate_id=candidate.candidate_id, case_id=case.case_id, repetition=repetition)
                    artifact = context.artifact_dir / hashlib.sha256(f"{case.case_id}:{repetition}:{number}".encode()).hexdigest()[:20]
                    artifact.mkdir(parents=True, exist_ok=True)
                    try:
                        if self.settings["mode"] == "smoke":
                            improved = "SMOKE_VERIFIED_INSTRUCTION" in (materialized.root / "instructions.md").read_text(encoding="utf-8")
                            result = {"score": float(improved), "valid_evaluation": True, "synthetic": True, "interactions": []}
                        else:
                            from .runner import run_case
                            result = await asyncio.to_thread(run_case, case, materialized.root, artifact, self.settings, context.run_id, "target_search")
                        trace = self.store.write_json(str((artifact / "trace.json").relative_to(self.store.run_dir)).replace('\\', '/'), result, kind="normalized-trace")
                        observation = Observation.create(candidate_id=candidate.candidate_id, case_id=case.case_id, split=context.split, repetition=repetition,
                            disposition="success" if result["score"] else "task_failure", score=result["score"], evaluator_fingerprint=self.fingerprint,
                            attempts=number, cost=Cost(rollouts=1), trace=trace, metadata={"synthetic": self.settings["mode"] == "smoke"})
                        context.attempt_sink.complete(attempt_id, observation, Cost(rollouts=1))
                        observations.append(observation)
                    except Exception as exc:
                        context.attempt_sink.fail(attempt_id, type(exc).__name__, Cost(rollouts=1))
                        raise  # Infrastructure errors must not masquerade as task failures.
        finally:
            materialized.release()
        return Evaluation(evaluation_id=sha256_digest(context.operation_id), candidate_id=candidate.candidate_id, split=context.split, purpose=context.purpose,
            iteration=context.iteration, requested_case_ids=tuple(c.case_id for c in cases), observations=tuple(observations),
            artifact_dir=ArtifactRef(uri=str(context.artifact_dir.relative_to(self.store.run_dir)).replace('\\', '/'), kind="evaluation-directory"))


def build_scenario(*, settings, base_dir, run_dir, store, ledger):
    benchmarks = selected_benchmarks(settings)
    repo, revision = prepare_seed()
    def verify(context):
        passed = bool((context.workspace / "instructions.md").read_text(encoding="utf-8").strip())
        # Both phases may change the complete target; phase describes intent.
        passed = passed and (context.workspace / "opencode-src/package.json").is_file()
        return GitVerificationVerdict(passed, "scope-and-instructions", "Candidate mutation scope checked", ())
    short_root = ROOT / ".test_tmp/ocw" / hashlib.sha256(str(run_dir).encode()).hexdigest()[:10]
    space = ShortGitHarnessSpace(short_root=short_root, source_repo=repo, base_revision=revision, store_root=run_dir / "candidates", worktree_root=short_root / "e", writable_paths=WRITABLE, verifier=verify)
    train, dev = [], []
    implementation = hashlib.sha256(b"".join(p.name.encode()+p.read_bytes() for p in sorted(PACKAGE.glob("*.py")))).hexdigest()
    entities = {"resolved/opencode-experiment.json": {"seed_revision": revision, "version": "1.19.5", "mode": settings["mode"], "adapter_implementation_sha256":implementation}}
    if settings.get("target_model", "").startswith("codex-acp/"):
        entities["resolved/opencode-experiment.json"]["acp_backend_instructions_sha256"] = sha256_digest((PACKAGE / "acp_gateway_native_instructions.md").read_bytes())
        entities["resolved/opencode-experiment.json"]["acp_adapter_sha256"] = sha256_digest((ROOT/"third_party/codex-acp/node_modules/@agentclientprotocol/codex-acp/dist/index.js").read_bytes())
    for bench in benchmarks:
        if settings["mode"] == "smoke":
            manifests = {"train": ["synthetic-train"], "development": ["synthetic-dev"]}
        else:
            from .case_selection import manifest_path
            path = manifest_path(settings, bench)
            manifests = json.loads(path.read_text())
            entities[f"resolved/{bench}-manifest.json"] = {"sha256": sha256_digest(path.read_bytes())}
        for split, bucket in (("train", train), ("development", dev)):
            from .case_selection import select_ids
            for task_id in select_ids(manifests[split], settings.get(f"{split}_per_benchmark", 100000), settings.get("case_selection_seed"), bench, split):
                bucket.append(Case(case_id=f"{bench}:{task_id}", split=split, payload={"benchmark": bench, "task_id": task_id}))
    phase_iterations = settings.get("capability_phase_iterations")
    if phase_iterations is None:
        per_benchmark = int(settings.get("batch_per_benchmark", 1))
        phase_iterations = int(settings.get("capability_phase_epochs", 1)) * max(math.ceil(sum(c.payload["benchmark"] == b for c in train) / per_benchmark) for b in benchmarks)
    prompt_class = OpenCodePromptPack
    if settings.get("opencode_prompt_version") == 1:
        from .opencode_prompt_pack import NativeOpenCodePromptPack
        prompt_class = NativeOpenCodePromptPack
    prompt_kwargs = {}
    if prompt_class is NativeOpenCodePromptPack and settings.get("optimizer_model_name"):
        prompt_kwargs["optimizer_model"] = settings["optimizer_model_name"]
    if settings.get("improved_autosaddler_v2"):
        if prompt_class is not NativeOpenCodePromptPack:
            raise ValueError("Improved AutoSaddler V2 requires the native full-source prompt pack")
        prompt_kwargs["adaptive_reflection_memory"] = True
    return ScenarioComponents(name="shared_opencode", version="0.1.0", harness_space=space, evaluator=JointEvaluator(space, store, settings),
        evidence_builder=MetaAREEvidenceBuilder(store=store),
        prompt_pack=prompt_class(store=store, writable_paths=WRITABLE, capability_phase_iterations=phase_iterations, **prompt_kwargs),
        train_cases=tuple(train), development_cases=tuple(dev), required_capabilities=CAPABILITIES, resolved_entities=entities)


class SmokeProvider:
    async def run(self, request):
        context = json.loads(request.spec.workspace_files[".autosaddler/session_context.json"])
        if request.spec.kind == "evolve":
            value = {"schema_version": "autosaddler-meta-are-evolution/v1", "parent_ids": [context["candidate_ids"][0]], "component_sources": {}, "rationale": "Synthetic setup check"}
        elif request.spec.kind == "diagnose_patch":
            path = request.workspace / "instructions.md"
            path.write_text(path.read_text(encoding="utf-8") + "\nSMOKE_VERIFIED_INSTRUCTION\n", encoding="utf-8")
            value = {"schema_version": "autosaddler-meta-are-diagnosis/v1", "intent": "Synthetic smoke patch", "diagnosis": "Synthetic check only", "expected_effect": "Synthetic score increase", "changed_paths": ["instructions.md"]}
        else:
            value = {"schema_version": "autosaddler-meta-are-reflection/v1", "lessons": []}
            if "causal_assessment" in request.spec.output_schema.get("required", []):
                value["causal_assessment"] = {"status": "uncertain", "behavioral_mechanism": "synthetic check",
                    "evidence": "scripted smoke outcomes only", "evidence_kind": "insufficient",
                    "evidence_case_ids": [], "source_refs": [], "next_step": "Use real traces in the campaign"}
        evidence = request.spec.workspace_files.get('.autosaddler/evidence_index.json')
        if evidence:
            value['evidence_receipt'] = json.loads(evidence)['files']
        return SessionResult(status="completed", structured_output=value, raw_response=json.dumps(value), tool_calls=(), usage=(), cost=Cost(sessions=1))


def build(config_path, run_id):
    from autosaddler.v2.config.models import RunConfig
    benchmarks = selected_benchmarks(RunConfig.load(config_path.resolve()).scenario.settings)
    registry = default_registry()
    registry.register_scenario(ScenarioRegistration(ScenarioPlugin("shared_opencode", "1", build_scenario), "builtin"))
    registry.providers["experiment_smoke"] = lambda **kwargs: SmokeProvider()
    from .optimizer_provider import create_provider
    registry.providers["opencode_acp"] = lambda **kwargs: create_provider(**kwargs, run_id=run_id)
    registry.task_selection["balanced_benchmarks"] = lambda **kwargs: BalancedSelection(**kwargs, benchmarks=benchmarks)
    registry.ranking["macro_benchmark_development"] = lambda: MacroRanking(benchmarks)
    runtime = build_runtime(config_path, run_id=run_id, registry=registry)
    if runtime.config.scenario.settings.get("require_complete_optimization_stages"):
        from .complete_engine import replace_engine
        runtime = replace_engine(runtime)
    return runtime
