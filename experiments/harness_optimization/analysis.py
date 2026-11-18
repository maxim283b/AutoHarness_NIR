"""Independent recomputation from raw JSONL and report generation."""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pilot_sql.tasks import load_tasks

from .engine import METHODS, atomic_json, read_jsonl


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    index = (len(sorted_values) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - index) + sorted_values[hi] * (index - lo)


def bootstrap_tasks(task_values: dict[str, list[float]], seed: int = 20261009, n_boot: int = 10000) -> list[float]:
    ids = sorted(task_values)
    if not ids:
        return [float("nan")] * 3
    point = _mean([_mean(task_values[task_id]) for task_id in ids])
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        chosen = [rng.choice(ids) for _ in ids]
        samples.append(_mean([_mean(task_values[task_id]) for task_id in chosen]))
    samples.sort()
    return [point, _percentile(samples, 0.025), _percentile(samples, 0.975)]


def _tail(rows: list[dict[str, Any]]) -> float:
    rewards = sorted(float(r["reward"]) for r in rows)
    n = max(1, math.ceil(0.20 * len(rewards)))
    return _mean(rewards[:n])


def _consumer_rows(events: list[dict[str, Any]], evaluations: dict[str, dict[str, Any]], phase: str) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("event") == "evaluation_used" and event.get("phase") == phase:
            grouped[(event["method"], int(event["algorithm_seed"]))].append(evaluations[event["cache_key"]])
    return grouped


def recompute_summary(run_dir: Path) -> dict[str, Any]:
    events = read_jsonl(run_dir / "optimization_log.jsonl")
    evaluation_rows = read_jsonl(run_dir / "evaluations.jsonl")
    evaluations = {row["cache_key"]: row for row in evaluation_rows}
    validation = _consumer_rows(events, evaluations, "validation")
    search = _consumer_rows(events, evaluations, "search")
    method_seed: list[dict[str, Any]] = []
    for (method, seed), rows in sorted(validation.items()):
        method_seed.append({
            "method": method, "algorithm_seed": seed, "n": len(rows),
            "success": _mean([float(r["success"]) for r in rows]),
            "reward": _mean([float(r["reward"]) for r in rows]),
            "tokens": _mean([float(r["tokens"]) for r in rows]),
            "target_calls": sum(int(r["target_calls"]) for r in rows),
            "proposer_calls": sum(int(r.get("proposer_calls", 0)) for r in rows),
            "tool_calls": sum(int(r["tool_calls"]) for r in rows),
            "latency_s": sum(float(r["latency_s"]) for r in rows),
            "tail_reward": _tail(rows),
            "worst_task_failure_rate": 1.0 - float(sorted(rows, key=lambda r: r["reward"])[0]["success"]),
            "candidate_hash": rows[0]["candidate_hash"] if rows else None,
            "context_budget": rows[0]["harness"]["context_budget"] if rows else None,
        })
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in method_seed:
        by_method[row["method"]].append(row)
    methods_summary: list[dict[str, Any]] = []
    for method, rows in sorted(by_method.items()):
        raw = [r for (m, _), rr in validation.items() if m == method for r in rr]
        reward_by_task: dict[str, list[float]] = defaultdict(list)
        success_by_task: dict[str, list[float]] = defaultdict(list)
        for item in raw:
            reward_by_task[item["task_id"]].append(float(item["reward"]))
            success_by_task[item["task_id"]].append(float(item["success"]))
        seed_rewards = sorted(r["reward"] for r in rows)
        search_rows = [r for (m, _), rr in search.items() if m == method for r in rr]
        seed_search = [r for (m, s), rr in search.items() if m == method for r in rr]
        base_search = [r for (m, _), rr in search.items() if m == "seed_harness" for r in rr]
        efficiency = (_mean([r["reward"] for r in seed_search]) - _mean([r["reward"] for r in base_search])) / max(1, len(search_rows)) if base_search else 0.0
        methods_summary.append({
            "method": method,
            "algorithm_seeds": len(rows),
            "validation_task_units": len(reward_by_task),
            "success": _mean([r["success"] for r in rows]),
            "reward": _mean([r["reward"] for r in rows]),
            "reward_task_bootstrap_ci95": bootstrap_tasks(reward_by_task),
            "success_task_bootstrap_ci95": bootstrap_tasks(success_by_task),
            "algorithm_seed_reward_interval": [_mean(seed_rewards), seed_rewards[0], seed_rewards[-1]],
            "tokens_per_task": _mean([r["tokens"] for r in rows]),
            "target_calls": sum(r["target_calls"] for r in rows),
            "proposer_calls": sum(r["proposer_calls"] for r in rows),
            "tool_calls": sum(r["tool_calls"] for r in rows),
            "tail_reward": _mean([r["tail_reward"] for r in rows]),
            "worst_task_failure_rate": _mean([r["worst_task_failure_rate"] for r in rows]),
            "unique_final_candidates": len({r["candidate_hash"] for r in rows}),
            "search_improvement_per_task_evaluation": efficiency,
        })
    candidates = [e for e in events if e.get("event") == "candidate_created"]
    rejected = sum(int(e.get("rejected_invalid_mutations", 0)) for e in candidates)
    unique_candidates = len({e["candidate_hash"] for e in candidates})
    pareto = []
    for row in methods_summary:
        dominated = any(
            other["success"] >= row["success"] and other["tokens_per_task"] <= row["tokens_per_task"]
            and (other["success"] > row["success"] or other["tokens_per_task"] < row["tokens_per_task"])
            for other in methods_summary
        )
        if not dominated:
            pareto.append({"method": row["method"], "success": row["success"], "tokens_per_task": row["tokens_per_task"]})
    ranked = sorted(methods_summary, key=lambda r: (-r["reward"], -r["success"], r["tokens_per_task"], r["method"]))
    co_best_methods: list[str] = []
    if ranked:
        leader = ranked[0]
        co_best_methods = [
            row["method"] for row in ranked
            if abs(row["reward"] - leader["reward"]) < 1e-12
            and abs(row["success"] - leader["success"]) < 1e-12
            and abs(row["tokens_per_task"] - leader["tokens_per_task"]) < 1e-12
        ]
    summary = {
        "status": "complete_core" if len(by_method) == len(METHODS) and all(len(v) == 3 for v in by_method.values()) else "partial",
        "held_out_opened": False,
        "primary_domain": "sql",
        "methods": methods_summary,
        "method_seed": method_seed,
        "best_method": ranked[0]["method"] if ranked else None,
        "co_best_methods": co_best_methods,
        "finalists": [r["method"] for r in ranked[:2]],
        "pareto_frontier": pareto,
        "unique_candidates": unique_candidates,
        "rejected_or_invalid_mutations": rejected,
        "raw_evaluation_records": len(evaluation_rows),
        "logical_search_evaluations": sum(1 for e in events if e.get("event") == "evaluation_used" and e.get("phase") == "search"),
        "logical_validation_evaluations": sum(1 for e in events if e.get("event") == "evaluation_used" and e.get("phase") == "validation"),
        "actual_target_calls": sum(int(e.get("target_calls", 0)) for e in events if e.get("event") == "evaluation_used"),
        "actual_proposer_calls": sum(int(e.get("proposer_calls", 0)) for e in events if e.get("event") == "evaluation_used"),
        "actual_tokens": sum(int(e.get("tokens", 0)) for e in events if e.get("event") == "evaluation_used" and not e.get("cache_hit")),
    }
    return summary


def audit(run_dir: Path, summary: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = summary or recompute_summary(run_dir)
    events = read_jsonl(run_dir / "optimization_log.jsonl")
    evaluations = read_jsonl(run_dir / "evaluations.jsonl")
    errors: list[str] = []
    warnings: list[str] = []
    keys = [r["cache_key"] for r in evaluations]
    if len(keys) != len(set(keys)):
        errors.append("duplicate evaluation cache keys")
    for row in evaluations:
        expected = float(row["success"]) - 0.10 * int(row["tokens"]) / 1000.0 - 0.02 * int(row["tool_calls"])
        if abs(expected - float(row["reward"])) > 1e-12:
            errors.append(f"reward mismatch: {row['cache_key']}")
        messages = json.dumps(row.get("model_messages", []), ensure_ascii=False).lower()
        if "reference_sql" in messages or "hidden_secret" in messages or "hidden test" in messages:
            errors.append(f"forbidden marker in model input: {row['cache_key']}")
    # Exact reference SQL may legitimately be the model output, but must never be in an input message.
    references = [task["reference_sql"].strip().lower() for task in load_tasks()]
    for row in evaluations:
        messages = json.dumps(row.get("model_messages", []), ensure_ascii=False).lower()
        if any(ref and ref in messages for ref in references):
            errors.append(f"exact reference SQL leaked to model input: {row['cache_key']}")
    first_validation = next((i for i, e in enumerate(events) if e.get("phase") == "validation"), None)
    if first_validation is not None:
        later_search_updates = [e for e in events[first_validation + 1:] if e.get("event") in {"candidate_created", "candidate_scored", "finalist_frozen"}]
        if later_search_updates:
            errors.append("optimizer update occurred after validation began")
    heldout = [r for r in evaluations if r.get("split") in {"heldout", "held-out", "test"}]
    if heldout:
        errors.append("held-out evaluations were opened")
    streams: dict[tuple[str, int], list[str]] = defaultdict(list)
    for e in events:
        if e.get("event") == "evaluation_used" and e.get("phase") == "search":
            streams[(e["method"], int(e["algorithm_seed"]))].append(e["task_id"])
    for seed in sorted({seed for _, seed in streams}):
        seqs = [seq for (method, s), seq in streams.items() if s == seed and method != "seed_harness"]
        if seqs and any(seq != seqs[0] for seq in seqs[1:]):
            errors.append(f"unequal search task stream for algorithm seed {seed}")
    expected_core = 30
    for key, seq in streams.items():
        if len(seq) != expected_core:
            warnings.append(f"logical search budget {key} is {len(seq)}, expected {expected_core}")
    model_ids = {r["model"]["model_id"] for r in evaluations}
    # `generation_max_tokens` and `context_budget` belong to the preregistered
    # harness genome.  They must participate in cache identity, but variation in
    # them is not a target-model switch.  Algorithm seed is likewise a planned
    # replication factor.  Audit only the frozen backend/request invariants here.
    model_invariant_fields = ("model_id", "endpoint", "temperature", "timeout_s", "artifact_hash")
    invariant_configs = {
        json.dumps({k: r["model"].get(k) for k in model_invariant_fields}, sort_keys=True)
        for r in evaluations
    }
    if len(model_ids) != 1:
        errors.append("more than one target model id")
    if len(invariant_configs) != 1:
        errors.append("inconsistent model configuration")
    if any(r["model"].get("artifact_hash") is None for r in evaluations):
        warnings.append("model artifact hash unavailable from LM Studio")
    required_eval_fields = {"timestamp", "cache_key", "candidate_hash", "task_id", "split", "seed",
                            "model_config_hash", "harness", "success", "reward", "tokens", "target_calls",
                            "proposer_calls", "tool_calls", "latency_s", "infrastructure_error"}
    for row in evaluations:
        missing = required_eval_fields - row.keys()
        if missing:
            errors.append(f"missing evaluation fields {sorted(missing)}: {row.get('cache_key')}")
    report = {
        "verdict": "reproducible" if not errors and not warnings else ("reproducible_with_caveats" if not errors else "not_reproducible"),
        "errors": sorted(set(errors)), "warnings": sorted(set(warnings)),
        "checks": {
            "unique_cache_keys": len(keys) == len(set(keys)),
            "reward_recomputed": not any("reward mismatch" in e for e in errors),
            "held_out_closed": not heldout,
            "no_validation_updates": not any("after validation" in e for e in errors),
            "model_ids": sorted(model_ids),
            "logical_search_evaluations": summary["logical_search_evaluations"],
            "logical_validation_evaluations": summary["logical_validation_evaluations"],
        },
    }
    return report


def write_reports(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = recompute_summary(run_dir)
    integrity = audit(run_dir, summary)
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "integrity_report.json", integrity)
    reports = ROOT / "reports"
    audit_dir = reports / "audit"
    reports.mkdir(exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Малоресурсная оптимизация фиксированного harness",
        "",
        f"Статус: `{summary['status']}`. Held-out — окончательно отложенная выборка — не открывался.",
        "",
        "## Validation результаты",
        "",
        "| метод | success | reward | task-bootstrap 95% CI | tokens/task | tail reward | worst failure |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary["methods"], key=lambda r: (-r["reward"], r["method"])):
        ci = row["reward_task_bootstrap_ci95"]
        lines.append(f"| `{row['method']}` | {row['success']:.3f} | {row['reward']:.3f} | [{ci[1]:.3f}, {ci[2]:.3f}] | {row['tokens_per_task']:.1f} | {row['tail_reward']:.3f} | {row['worst_task_failure_rate']:.3f} |")
    co_best = summary.get("co_best_methods", [])
    if "seed_harness" in co_best:
        conclusion = (
            "Автоматическая оптимизация не улучшила seed harness на validation: "
            f"одинаковый лучший результат разделили {', '.join('`'+x+'`' for x in co_best)}. "
            f"Поле `best_method={summary['best_method']}` — лишь детерминированный порядок внутри этой ничьей, а не доказательство превосходства."
        )
    else:
        conclusion = f"Лучший средний validation reward показал `{summary['best_method']}`."
    lines += [
        "", "## Бюджеты", "",
        f"- Логических search task-evaluations: {summary['logical_search_evaluations']}.",
        f"- Логических validation task-evaluations: {summary['logical_validation_evaluations']}.",
        f"- Фактических target calls после cache: {summary['actual_target_calls']}.",
        f"- Proposer calls: {summary['actual_proposer_calls']}.",
        f"- Фактических токенов: {summary['actual_tokens']}.",
        "", "## Вывод", "",
        conclusion,
        f"Финалисты по preregistered правилу: {', '.join('`'+x+'`' for x in summary['finalists'])}.",
        "Tail-оптимизация не улучшила нижний хвост относительно seed harness; усложнённый verifier loop увеличил расход токенов и вызовов, не повысив success.",
        "Интервалы bootstrap пересэмплируют задачи; диапазоны по algorithm seeds в summary имеют другую интерпретацию и не являются CI переноса на новые задачи.",
        "", "## Ограничения и отклонения", "",
        "- Артефактный hash модели LM Studio недоступен; фиксирован только model id и параметры запроса.",
        "- `seed_harness` не выполняет поиск кандидатов и получил 10 search task-evaluations на seed против 30 у оптимизаторов; это отмечено независимым аудитом как асимметрия бюджета.",
        "- Core заранее ограничен тремя кандидатами на seed, а не рекомендованными 25 кандидатами; это зафиксировано до validation.",
        "- Context 4096, режим 25 observations, перенос на Python, strong-model reference и held-out оставлены optional extension.",
        "- `literature_inspired_baseline` не является точным AutoSaddler/AutoHarness reproduction.",
        "- `core_20261009_v1` прерван во время разработки resumability и исключён; его append-only сырые логи не переписывались. Основной результат — отдельный `core_20261009_v2`.",
        "- Первичный audit ошибочно трактовал оптимизируемый `generation_max_tokens` как смену модели. Исправлен только пост-hoc анализ с регрессионным тестом; raw JSONL `v2` не изменялся.",
        "", "## Воспроизведение", "",
        "```powershell",
        "$env:TEMP=(Resolve-Path '.\\.test_tmp').Path",
        "$env:TMP=$env:TEMP",
        "& '.\\pilot\\.venv\\Scripts\\python.exe' -m experiments.harness_optimization.cli audit --run-dir '" + str(run_dir).replace("\\", "\\\\") + "'",
        "```",
    ]
    (reports / "harness_optimization.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit_lines = [
        "# Независимый аудит harness optimization", "",
        f"Вердикт: `{integrity['verdict']}`.", "",
        "Пересчёт выполнен из `optimization_log.jsonl` и `evaluations.jsonl`, без использования сохранённого summary как источника истины.",
        "", "## Ошибки", "",
        *(f"- {item}" for item in integrity["errors"]),
        *( ["- Нет."] if not integrity["errors"] else [] ),
        "", "## Предупреждения", "",
        *(f"- {item}" for item in integrity["warnings"]),
        *( ["- Нет."] if not integrity["warnings"] else [] ),
        "", "## Проверки", "",
        "```json", json.dumps(integrity["checks"], ensure_ascii=False, indent=2), "```",
    ]
    (audit_dir / "harness_optimization_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    return summary, integrity
