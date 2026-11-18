"""Search engine, LM Studio evaluator, append-only logs, cache, and resume."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pilot_common.adapters import SQLTaskAdapter
from pilot_sql.tasks import load_tasks
from pilot_sql.verifier import verify

from .genome import MUTATION_OPERATORS, HarnessGenome, canonical_json, content_hash, mutate_genome, seed_genome


METHODS = (
    "seed_harness",
    "random_mutation_search",
    "literature_inspired_baseline",
    "bandit_mutation_operator",
    "sequential_evolution_mean",
    "sequential_evolution_tail",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def immutable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


@dataclass(frozen=True)
class ModelSettings:
    model_id: str = "ministral-3-8b-instruct-2512"
    endpoint: str = "http://127.0.0.1:1234/v1/chat/completions"
    temperature: float = 0.0
    seed: int = 20261009
    context_budget: int = 2048
    generation_max_tokens: int = 384
    timeout_s: float = 180.0
    artifact_hash: str | None = None

    @property
    def config_hash(self) -> str:
        return content_hash(asdict(self))


class LMStudioClient:
    def __init__(self, settings: ModelSettings):
        self.settings = settings

    def chat(self, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
        body = {
            "model": self.settings.model_id,
            "messages": messages,
            "temperature": self.settings.temperature,
            "seed": self.settings.seed,
            "max_tokens": max_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            self.settings.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.settings.timeout_s) as response:
            output = json.loads(response.read())
        elapsed = time.perf_counter() - started
        usage = output.get("usage") or {}
        choice = (output.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        return {
            "text": text,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "latency_s": elapsed,
            "finish_reason": choice.get("finish_reason"),
        }


def _extract_sql(text: str) -> str:
    text = (text or "").strip()
    if "```" in text:
        pieces = text.split("```")
        if len(pieces) >= 3:
            text = pieces[1]
            if text.lstrip().lower().startswith("sql"):
                text = text.lstrip()[3:]
    upper = text.upper()
    starts = [i for i in (upper.find("SELECT"), upper.find("WITH")) if i >= 0]
    if starts:
        text = text[min(starts):]
    return text.strip().rstrip(";").strip()


def _fit_context(system: str, user: str, budget: int) -> tuple[str, str]:
    # Conservative deterministic approximation for local OpenAI-compatible APIs.
    char_budget = max(512, budget * 3)
    if len(system) + len(user) <= char_budget:
        return system, user
    keep = max(256, char_budget - len(system))
    return system, user[:keep]


def _planning_instruction(mode: str) -> str:
    return {
        "none": "",
        "concise_private_plan": " Privately form a concise plan before producing SQL.",
        "checklist": " Privately check schema names, filters, aggregation, and ordering before producing SQL.",
    }[mode]


def evaluate_sql_task(
    task: dict[str, Any], genome: HarnessGenome, client: Any, settings: ModelSettings
) -> dict[str, Any]:
    genome.validate()
    adapter = SQLTaskAdapter()
    public = adapter.public_task(task)
    system = genome.system_prompt + _planning_instruction(genome.planning_mode)
    base_user = adapter.initial_user(task, type("Cfg", (), {"use_visible_verifier": genome.use_visible_verifier})())
    calls: list[dict[str, Any]] = []
    verifier_events: list[dict[str, Any]] = []

    def ask(name: str, user: str) -> str:
        fitted_system, fitted_user = _fit_context(system, user, genome.context_budget)
        messages = [{"role": "system", "content": fitted_system}, {"role": "user", "content": fitted_user}]
        response = client.chat(messages, genome.generation_max_tokens)
        calls.append({"name": name, "messages": messages, **response})
        return _extract_sql(response["text"])

    if genome.pattern == "plan_execute":
        plan = ask("plan", base_user + "\n\nGive a short reasoning plan only; do not output SQL yet.")
        sql = ask("execute", base_user + f"\n\nPrivate plan from the previous step:\n{plan}\n\nNow return only the SQL query.")
    else:
        sql = ask("solve", base_user)
        if genome.pattern == "self_refine":
            sql = ask("self_refine", adapter.refine_user(task, sql))

    result = None
    if genome.pattern == "verifier_loop":
        for attempt in range(genome.max_repairs + 1):
            result = verify(task, sql)
            verifier_events.append({k: result.get(k) for k in ("success", "error", "error_code", "elapsed_s", "row_count")})
            if result["success"] or attempt >= genome.max_repairs:
                break
            generic_error = result.get("error") if genome.use_visible_verifier else "The query failed verification."
            sql = ask(f"repair_{attempt + 1}", adapter.repair_user(task, sql, generic_error))
    if result is None:
        result = verify(task, sql)
        verifier_events.append({k: result.get(k) for k in ("success", "error", "error_code", "elapsed_s", "row_count")})

    tokens = sum(int(c.get("prompt_tokens", 0)) + int(c.get("completion_tokens", 0)) for c in calls)
    tool_calls = len(verifier_events)
    reward = float(bool(result["success"])) - 0.10 * tokens / 1000.0 - 0.02 * tool_calls
    return {
        "task_id": task["task_id"],
        "task_template": task["template_group"],
        "public_task": public,
        "success": bool(result["success"]),
        "reward": reward,
        "tokens": tokens,
        "prompt_tokens": sum(int(c.get("prompt_tokens", 0)) for c in calls),
        "completion_tokens": sum(int(c.get("completion_tokens", 0)) for c in calls),
        "target_calls": len(calls),
        "proposer_calls": 0,
        "tool_calls": tool_calls,
        "latency_s": sum(float(c.get("latency_s", 0.0)) for c in calls) + sum(float(v.get("elapsed_s", 0.0)) for v in verifier_events),
        "error_code": result.get("error_code"),
        "final_output": sql,
        "model_messages": [c["messages"] for c in calls],
        "model_call_summaries": [{k: c.get(k) for k in ("name", "prompt_tokens", "completion_tokens", "latency_s", "finish_reason")} for c in calls],
        "verifier_events": verifier_events,
    }


def objective(rows: list[dict[str, Any]], kind: str) -> float:
    rewards = [float(r["reward"]) for r in rows]
    mean = sum(rewards) / len(rewards)
    if kind == "mean":
        return mean
    n_tail = max(1, math.ceil(0.20 * len(rewards)))
    tail = sum(sorted(rewards)[:n_tail]) / n_tail
    return 0.5 * mean + 0.5 * tail


class ExperimentRun:
    def __init__(self, run_dir: Path, protocol: dict[str, Any], client_factory: Callable[[ModelSettings], Any] = LMStudioClient):
        self.run_dir = run_dir
        self.protocol = protocol
        backend = protocol["backend"]
        self.base_settings = ModelSettings(
            model_id=backend["model_id"], endpoint=backend["chat_endpoint"],
            temperature=backend["temperature"], seed=backend["generation_seed"],
            context_budget=backend["context_size"], generation_max_tokens=backend["generation_max_tokens"],
            artifact_hash=backend.get("model_artifact_hash"),
        )
        self.client_factory = client_factory
        self.opt_log = run_dir / "optimization_log.jsonl"
        self.eval_log = run_dir / "evaluations.jsonl"
        self.cache = {r["cache_key"]: r for r in read_jsonl(self.eval_log)}
        self.event_keys = {self._event_key(r) for r in read_jsonl(self.opt_log)}
        manifest = json.loads((ROOT / "pilot_sql" / "splits_manifest.json").read_text(encoding="utf-8"))
        by_id = {t["task_id"]: t for t in load_tasks()}
        self.tasks = {split: [by_id[task_id] for task_id in ids] for split, ids in manifest["splits"].items()}

    @staticmethod
    def _event_key(record: dict[str, Any]) -> tuple[Any, ...]:
        """Stable logical identity used to make append-only resume idempotent."""
        kind = record.get("event")
        if kind == "evaluation_used":
            return (kind, record.get("phase"), record.get("method"), record.get("algorithm_seed"),
                    record.get("cache_key"))
        if kind in {"candidate_created", "candidate_scored"}:
            return (kind, record.get("method"), record.get("algorithm_seed"), record.get("candidate_index"),
                    record.get("candidate_hash"))
        if kind in {"finalist_frozen", "validation_complete"}:
            return (kind, record.get("method"), record.get("algorithm_seed"), record.get("candidate_hash"))
        if kind == "all_finalists_frozen":
            return (kind, tuple(record.get("methods", [])), tuple(record.get("seeds", [])))
        if kind == "infrastructure_stop":
            return (kind, record.get("phase"), record.get("method"), record.get("algorithm_seed"),
                    record.get("candidate_hash"), record.get("task_id"))
        return (kind, record.get("timestamp"))

    def _event(self, **record: Any) -> None:
        key = self._event_key(record)
        if key in self.event_keys:
            return
        append_jsonl(self.opt_log, {"timestamp": now_iso(), **record})
        self.event_keys.add(key)

    def _save_candidate(self, genome: HarnessGenome) -> None:
        immutable_json(self.run_dir / "candidates" / f"{genome.candidate_hash}.json", genome.to_dict())

    def evaluate(self, genome: HarnessGenome, task: dict[str, Any], split: str, method: str,
                 algorithm_seed: int, phase: str) -> dict[str, Any]:
        settings = ModelSettings(**{**asdict(self.base_settings), "seed": self.base_settings.seed + algorithm_seed,
                                   "context_budget": genome.context_budget,
                                   "generation_max_tokens": genome.generation_max_tokens})
        key = content_hash({"candidate_hash": genome.candidate_hash, "task_id": task["task_id"],
                            "seed": settings.seed, "model_config_hash": settings.config_hash})
        if key in self.cache:
            row = self.cache[key]
            cache_hit = True
        else:
            client = self.client_factory(settings)
            failures = []
            for attempt in range(3):
                try:
                    result = evaluate_sql_task(task, genome, client, settings)
                    break
                except Exception as exc:
                    failures.append({"type": type(exc).__name__, "message": str(exc)})
                    if attempt == 2:
                        self._event(event="infrastructure_stop", phase=phase, method=method,
                                    algorithm_seed=algorithm_seed, candidate_hash=genome.candidate_hash,
                                    task_id=task["task_id"], infrastructure_errors=failures)
                        raise
                    time.sleep(1 + attempt)
            row = {
                "timestamp": now_iso(), "cache_key": key, "candidate_hash": genome.candidate_hash,
                "task_id": task["task_id"], "split": split, "seed": settings.seed,
                "model_config_hash": settings.config_hash, "model": asdict(settings),
                "harness": genome.to_dict(), "infrastructure_error": None, **result,
            }
            append_jsonl(self.eval_log, row)
            self.cache[key] = row
            cache_hit = False
        self._event(
            event="evaluation_used", phase=phase, method=method, algorithm_seed=algorithm_seed,
            candidate_hash=genome.candidate_hash, task_id=task["task_id"], split=split,
            cache_key=key, cache_hit=cache_hit, success=row["success"], reward=row["reward"],
            tokens=row["tokens"], target_calls=0 if cache_hit else row["target_calls"],
            proposer_calls=0, tool_calls=row["tool_calls"], latency_s=0.0 if cache_hit else row["latency_s"],
        )
        return row

    def _candidate_rows(self, genome: HarnessGenome, tasks: list[dict[str, Any]], split: str,
                        method: str, algorithm_seed: int, phase: str) -> list[dict[str, Any]]:
        return [self.evaluate(genome, task, split, method, algorithm_seed, phase) for task in tasks]

    def _completed_final(self, method: str, seed: int) -> dict[str, Any] | None:
        path = self.run_dir / "final_configs" / f"{method}_seed{seed}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def optimize_method(self, method: str, algorithm_seed: int, n_tasks: int = 10, n_candidates: int = 3) -> HarnessGenome:
        if method not in METHODS:
            raise ValueError(method)
        existing = self._completed_final(method, algorithm_seed)
        if existing:
            return HarnessGenome(**existing["genome"])
        rng = random.Random(20261009 + algorithm_seed * 1009 + METHODS.index(method) * 7919)
        tasks = self.tasks["search"][:n_tasks]
        population: list[tuple[HarnessGenome, float]] = []
        champion = seed_genome(self.base_settings.context_budget)
        bandit = {op: {"n": 0, "value": 0.0} for op in MUTATION_OPERATORS}
        objective_kind = "tail" if method == "sequential_evolution_tail" else "mean"
        used_hashes: set[str] = set()
        cumulative = {"task_evaluations": 0, "target_calls": 0, "proposer_calls": 0, "tokens": 0}
        for index in range(n_candidates):
            parent = champion
            operator = "seed"
            rejected = 0
            if index == 0 or method == "seed_harness":
                candidate = seed_genome(self.base_settings.context_budget)
            else:
                if method == "random_mutation_search":
                    parent = seed_genome(self.base_settings.context_budget)
                    operator = rng.choice(MUTATION_OPERATORS)
                elif method == "literature_inspired_baseline":
                    operator = MUTATION_OPERATORS[(index - 1) % len(MUTATION_OPERATORS)]
                elif method == "bandit_mutation_operator":
                    unseen = [op for op, state in bandit.items() if state["n"] == 0]
                    if unseen:
                        operator = unseen[(index - 1) % len(unseen)]
                    else:
                        total = sum(s["n"] for s in bandit.values())
                        operator = max(MUTATION_OPERATORS, key=lambda op: bandit[op]["value"] + math.sqrt(2 * math.log(total) / bandit[op]["n"]))
                else:
                    if population:
                        top = sorted(population, key=lambda item: (-item[1], item[0].candidate_hash))[:2]
                        parent = rng.choice(top)[0]
                    operator = rng.choice(MUTATION_OPERATORS)
                for retry in range(len(MUTATION_OPERATORS) + 2):
                    try:
                        candidate = mutate_genome(parent, operator, algorithm_seed * 10000 + index * 100 + retry)
                        if candidate.candidate_hash not in used_hashes:
                            break
                        rejected += 1
                        operator = MUTATION_OPERATORS[(MUTATION_OPERATORS.index(operator) + 1) % len(MUTATION_OPERATORS)]
                    except ValueError:
                        rejected += 1
                        operator = MUTATION_OPERATORS[(MUTATION_OPERATORS.index(operator) + 1) % len(MUTATION_OPERATORS)]
                else:
                    raise RuntimeError("could not create a unique valid mutation")
            candidate.validate()
            self._save_candidate(candidate)
            used_hashes.add(candidate.candidate_hash)
            self._event(event="candidate_created", phase="search", method=method, algorithm_seed=algorithm_seed,
                        candidate_index=index, candidate_hash=candidate.candidate_hash,
                        parent_hash=None if index == 0 else parent.candidate_hash,
                        mutation_operator=operator, rejected_invalid_mutations=rejected,
                        harness=candidate.to_dict(), optimizer_state={"bandit": bandit, "population_size": len(population)},
                        cumulative_budget=dict(cumulative))
            rows = self._candidate_rows(candidate, tasks, "search", method, algorithm_seed, "search")
            score = objective(rows, objective_kind)
            cumulative["task_evaluations"] += len(rows)
            cumulative["target_calls"] += sum(r["target_calls"] for r in rows)
            cumulative["tokens"] += sum(r["tokens"] for r in rows)
            accepted = not population or score > max(item[1] for item in population)
            population.append((candidate, score))
            if method in {"literature_inspired_baseline", "bandit_mutation_operator"} and accepted:
                champion = candidate
            if method == "bandit_mutation_operator" and operator != "seed":
                previous = max((s for _, s in population[:-1]), default=score)
                state = bandit[operator]
                state["n"] += 1
                state["value"] += ((score - previous) - state["value"]) / state["n"]
            self._event(event="candidate_scored", phase="search", method=method, algorithm_seed=algorithm_seed,
                        candidate_index=index, candidate_hash=candidate.candidate_hash,
                        parent_hash=None if index == 0 else parent.candidate_hash, mutation_operator=operator,
                        objective_kind=objective_kind, objective=score, accepted=accepted,
                        reason="improved_search_objective" if accepted else "not_better_than_champion",
                        optimizer_state={"bandit": bandit, "scores": [s for _, s in population]},
                        cumulative_budget=dict(cumulative))
        finalist, finalist_score = sorted(population, key=lambda item: (-item[1], item[0].candidate_hash))[0]
        frozen = {"method": method, "algorithm_seed": algorithm_seed, "candidate_hash": finalist.candidate_hash,
                  "search_objective": finalist_score, "objective_kind": objective_kind,
                  "genome": finalist.to_dict(), "frozen_at": now_iso(), "cumulative_budget": cumulative}
        immutable_json(self.run_dir / "final_configs" / f"{method}_seed{algorithm_seed}.json", frozen)
        self._event(event="finalist_frozen", phase="freeze", method=method, algorithm_seed=algorithm_seed,
                    candidate_hash=finalist.candidate_hash, search_objective=finalist_score,
                    objective_kind=objective_kind, cumulative_budget=dict(cumulative))
        return finalist

    def validate_method(self, method: str, algorithm_seed: int, genome: HarnessGenome, n_tasks: int = 10) -> None:
        marker = self.run_dir / "final_configs" / f"{method}_seed{algorithm_seed}.validation_complete"
        if marker.exists():
            return
        rows = self._candidate_rows(genome, self.tasks["validation"][:n_tasks], "validation", method,
                                    algorithm_seed, "validation")
        immutable_json(marker, {"candidate_hash": genome.candidate_hash, "n": len(rows), "completed_at": now_iso()})
        self._event(event="validation_complete", phase="validation", method=method,
                    algorithm_seed=algorithm_seed, candidate_hash=genome.candidate_hash, n=len(rows))

    def run_matrix(self, methods: tuple[str, ...] = METHODS, seeds: tuple[int, ...] = (0, 1, 2),
                   n_tasks: int = 10, n_candidates: int = 3, validation_tasks: int = 10) -> None:
        # Freeze every candidate before any new validation result is obtained.
        finalists: dict[tuple[str, int], HarnessGenome] = {}
        for seed in seeds:
            for method in methods:
                finalists[(method, seed)] = self.optimize_method(method, seed, n_tasks, n_candidates)
        self._event(event="all_finalists_frozen", phase="freeze", methods=list(methods), seeds=list(seeds))
        for seed in seeds:
            for method in methods:
                self.validate_method(method, seed, finalists[(method, seed)], validation_tasks)


def create_manifest(run_dir: Path, protocol_path: Path, run_id: str, mode: str) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = {
        "run_id": run_id, "created_at": now_iso(), "mode": mode,
        "protocol_path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": content_hash(protocol), "protocol": protocol,
        "held_out_opened": False, "raw_logs_append_only": True,
        "code_files": {},
    }
    for path in sorted(Path(__file__).parent.glob("*.py")):
        manifest["code_files"][str(path.relative_to(ROOT)).replace("\\", "/")] = content_hash(path.read_text(encoding="utf-8"))
    immutable_json(run_dir / "manifest.json", manifest)
    return manifest
