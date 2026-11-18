"""Exploratory stage-2 online confirmation for the two frozen finalists.

The task stream is the five SQL `online` tasks that were not part of the
40-task signal experiment.  Held-out tasks are rejected.  For each policy and
task exactly one selected harness is executed; no counterfactual harness is
queried.  The append-only raw journal provides resume, and the final gzip log
is rebuilt deterministically from it after integrity checks.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(os.path.abspath(__file__)).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pilot"))

import policy_selection as ps  # noqa: E402
from pilot.sql_signal_collect import (  # noqa: E402
    PublicMemoryBank, collector_lock, leakage_errors, make_config, public_record, source_hash,
)
from pilot_common import OpenAICompatibleClient, SQLTaskAdapter, run_task  # noqa: E402
from pilot_sql.tasks import dataset_hash, load_tasks  # noqa: E402
from pilot_sql.verifier import verify  # noqa: E402


ONLINE_SEED = 20261011
REPLAY_SEED = 0
REPLAY_BUDGET = 50
SCOPE = "combined"
FINALISTS = (ps.LinUCB, ps.LinearThompson)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def new_online_tasks():
    manifest = json.loads((ROOT / "pilot_sql" / "splits_manifest.json").read_text(encoding="utf-8"))
    old = set(manifest["first_experiment_ids"])
    heldout = set(manifest["splits"]["held-out"])
    ids = [x for x in manifest["splits"]["online"] if x not in old]
    if set(ids) & heldout or set(ids) & old:
        raise RuntimeError("new online stream overlaps old experiment or held-out")
    if ids != ["sql-07-02", "sql-08-03", "sql-08-04", "sql-10-03", "sql-10-04"]:
        raise RuntimeError(f"unexpected new online ids: {ids}")
    by_id = {x["task_id"]: x for x in load_tasks()}
    return [by_id[x] for x in ids], manifest


def restore_policy(P, units, encoder):
    train, _ = ps.pools(units, SCOPE)
    perm = list(train)
    random.Random([ps.BASE_SEED, REPLAY_SEED, len(SCOPE)].__repr__()).shuffle(perm)
    rng = np.random.default_rng([ps.BASE_SEED, REPLAY_SEED, REPLAY_BUDGET,
                                 ps.POLICIES.index(P), len(SCOPE)])
    policy = P(rng, len(encoder(train[0])))
    for t, unit in enumerate(perm[:REPLAY_BUDGET]):
        x = encoder(unit)
        arm = policy.select(x, t, deploy=False)
        outcome = unit["outcomes"][ps.HARNESSES[arm]]
        policy.update(arm, outcome["reward"], x)
    return policy


def validate_frozen_configs():
    frozen = ROOT / "results" / "policy" / "frozen_finalists"
    files = sorted(frozen.glob("*.json"))
    names = [json.loads(x.read_text(encoding="utf-8"))["policy"] for x in files]
    expected = [P.name for P in FINALISTS]
    if names != expected:
        raise RuntimeError(f"frozen finalists changed: {names}, expected {expected}")
    return {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in files}


def result_record(policy_name, task, pattern, trajectory, cfg, policy_state, t):
    record = public_record(task, "online_new", ONLINE_SEED, pattern, trajectory, cfg)
    record.update({
        "phase": "online", "policy": policy_name, "online_t": t,
        "reward": ps.reward(record), "policy_state_after_update": policy_state,
        "counterfactual_harnesses_executed": [],
    })
    return record


def update_main_artifacts(rows, manifest, integrity):
    out = ROOT / "results" / "policy"
    with gzip.open(out / "online_log.jsonl.gz", "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    by_policy = {}
    for P in FINALISTS:
        rr = [x for x in rows if x["policy"] == P.name]
        by_policy[P.name] = {
            "n_tasks": len(rr), "mean_reward": ps.mean([x["reward"] for x in rr]),
            "success": ps.mean([float(x["success"]) for x in rr]),
            "mean_tokens": ps.mean([float(x["tokens"]) for x in rr]),
            "mean_tool_calls": ps.mean([float(x["tool_calls"]) for x in rr]),
            "selection_counts": dict(Counter(x["harness"] for x in rr)),
        }
    online_summary = {
        "status": "complete_exploratory_user_authorized_below_registered_gate",
        "manifest_hash": canonical_hash(manifest), "seed": ONLINE_SEED,
        "replay_initialization": {"scope": SCOPE, "seed": REPLAY_SEED, "budget": REPLAY_BUDGET},
        "task_ids": manifest["task_ids"], "heldout_overlap": [], "per_policy": by_policy,
        "integrity": integrity,
    }
    (out / "online_summary.json").write_text(json.dumps(online_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["stage2_status"] = online_summary["status"]
    summary["online"] = online_summary
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = ROOT / "reports" / "policy_comparison.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace(
        "Конфигурации не более двух финалистов заморожены в `results/policy/frozen_finalists/`. Реальный новый LM Studio поток не был подменён replay: ни один финалист не прошёл предзарегистрированный gate, а нового task manifest нет. `online_log.jsonl.gz` содержит только явную status-запись без task labels; подробные события replay находятся в `replay_log.jsonl.gz`.",
        "Конфигурации двух финалистов были заморожены после replay. Затем по прямому решению пользователя выполнен отдельный exploratory online-запуск, несмотря на непройденный предзарегистрированный gate; его результаты приведены ниже."
    )
    marker = "\n## Фактический exploratory online-запуск\n"
    if marker in report:
        report = report.split(marker)[0].rstrip() + "\n"
    lines = ["", "## Фактический exploratory online-запуск", "",
             "Запуск выполнен по прямому решению пользователя, несмотря на непройденный предзарегистрированный gate. "
             "Использованы пять новых SQL online-задач; held-out не открывался. На каждой задаче каждая policy запускала только выбранный harness.", "",
             "| policy | n | reward | success | tokens | tools | выборы |", "|---|---:|---:|---:|---:|---:|---|"]
    for name, x in by_policy.items():
        picks = ", ".join(f"{k}: {v}" for k, v in sorted(x["selection_counts"].items()))
        lines.append(f"| `{name}` | {x['n_tasks']} | {x['mean_reward']:.3f} | {x['success']:.3f} | "
                     f"{x['mean_tokens']:.1f} | {x['mean_tool_calls']:.2f} | {picks} |")
    lines += ["", "Воспроизведение online-этапа с автоматическим resume:", "", "```powershell",
              "$env:PILOT_BACKEND='lmstudio'", "$env:PILOT_MODEL='ministral-3-8b-instruct-2512'",
              "python pilot/policy_online.py", "```"]
    report_path.write_text(report.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    integrity_path = out / "integrity_report.json"
    overall = json.loads(integrity_path.read_text(encoding="utf-8"))
    overall["online_stage"] = online_summary
    tracked = [out / "online_log.jsonl.gz", out / "replay_log.jsonl.gz", out / "online_raw.jsonl",
               out / "online_manifest.json", out / "online_summary.json", out / "per_seed.jsonl",
               out / "summary.json"]
    overall["outputs"] = {str(path.relative_to(ROOT)): file_hash(path) for path in tracked}
    integrity_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results" / "policy")
    args = ap.parse_args(argv)
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frozen_hashes = validate_frozen_configs()
    tasks, split_manifest = new_online_tasks()
    all_tasks = load_tasks()
    references = [x["reference_sql"] for x in all_tasks]
    memory = PublicMemoryBank([x for x in all_tasks if x["task_id"] in split_manifest["splits"]["search"]])
    manifest = {
        "dataset": "pilot_sql", "dataset_hash": dataset_hash(), "split": "online",
        "selection_rule": "online ids not included in frozen first_experiment_ids, manifest order",
        "task_ids": [x["task_id"] for x in tasks], "old_experiment_overlap": [],
        "heldout_overlap": [], "frozen_finalist_hashes": frozen_hashes,
        "model": "ministral-3-8b-instruct-2512", "temperature": 0.0, "seed": ONLINE_SEED,
    }
    manifest_path = out / "online_manifest.json"
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2)
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != rendered:
        raise RuntimeError("refusing to change frozen online manifest")
    if not manifest_path.exists():
        manifest_path.write_text(rendered, encoding="utf-8")
    raw_path = out / "online_raw.jsonl"
    existing = read_jsonl(raw_path)
    existing_by = {(x["policy"], x["task_id"]): x for x in existing}
    if len(existing_by) != len(existing):
        raise RuntimeError("duplicate online raw rows")
    units, _ = ps.load_units()
    train, validation = ps.pools(units, SCOPE)
    encoder = ps.Encoder(train + validation, cross_domain=True)
    hsource = source_hash()
    client = None
    policies = {P.name: restore_policy(P, units, encoder) for P in FINALISTS}
    # Resume reconstructs policy state only from that policy's actually observed outcomes.
    for P in FINALISTS:
        pol = policies[P.name]
        for task in tasks:
            old = existing_by.get((P.name, task["task_id"]))
            if old:
                pol.update(ps.HARNESSES.index(old["harness"]), float(old["reward"]), encoder({
                    "domain": "sql", "features": old["task_features"]}))
    lock_root = ROOT / "runs"
    with collector_lock(lock_root):
        for t, task in enumerate(tasks):
            for P in FINALISTS:
                if (P.name, task["task_id"]) in existing_by:
                    continue
                pol = policies[P.name]
                unit = {"domain": "sql", "features": {
                    **task.get("features", {}), "n_words": len(task["description"].split()),
                    "n_chars": len(task["description"]),
                    "n_allowed_operations": len(task.get("allowed_operations", [])),
                }}
                x = encoder(unit)
                arm = pol.select(x, REPLAY_BUDGET + t, deploy=False)
                pattern = ps.HARNESSES[arm]
                cfg = make_config(pattern, ONLINE_SEED, hsource)
                if client is None:
                    client = OpenAICompatibleClient(cfg.model)
                trajectory = run_task(task, cfg, client, lambda answer, current=task: verify(current, answer),
                                      memory=memory, adapter=SQLTaskAdapter())
                if trajectory.error_type:
                    raise RuntimeError(f"LM Studio infrastructure error: {trajectory.error_type}")
                provisional = public_record(task, "online_new", ONLINE_SEED, pattern, trajectory, cfg)
                leaks = leakage_errors(provisional, references)
                if leaks:
                    raise RuntimeError("leakage guard failed: " + "; ".join(leaks))
                r = ps.reward(provisional)
                pol.update(arm, r, x)
                record = result_record(P.name, task, pattern, trajectory, cfg, pol.state(), t)
                with raw_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                existing.append(record)
                existing_by[(P.name, task["task_id"])] = record
                print(f"[{len(existing)}/{len(tasks) * len(FINALISTS)}] {P.name} {task['task_id']} "
                      f"{pattern} success={record['success']} reward={record['reward']:.3f}", flush=True)
    expected = {(P.name, t["task_id"]) for P in FINALISTS for t in tasks}
    actual = set(existing_by)
    integrity = {
        "expected_rows": len(expected), "completed_rows": len(actual & expected),
        "missing": [list(x) for x in sorted(expected - actual)],
        "unexpected": [list(x) for x in sorted(actual - expected)],
        "duplicate_rows": len(existing) - len(existing_by), "heldout_rows": 0,
        "counterfactual_executions_logged": sum(len(x["counterfactual_harnesses_executed"]) for x in existing),
        "policy_state_after_every_update": all("policy_state_after_update" in x for x in existing),
        "infrastructure_errors": sum(bool(x.get("infrastructure_error")) for x in existing),
        "leakage_errors": sum((leakage_errors(x, references) for x in existing), []),
    }
    if integrity["missing"] or integrity["unexpected"] or integrity["duplicate_rows"] or integrity["leakage_errors"]:
        raise RuntimeError(f"online integrity failure: {integrity}")
    update_main_artifacts(existing, manifest, integrity)
    print("online stage complete")


if __name__ == "__main__":
    main()
