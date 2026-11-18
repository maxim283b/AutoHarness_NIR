"""Collect the pre-registered SQL harness-signal matrix through LM Studio.

The orchestration in this module is experiment-specific.  Pattern execution and
SQL prompt construction remain delegated to :mod:`pilot_common`; correctness is
always decided by :func:`pilot_sql.verifier.verify`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(ROOT))

from pilot_common import (  # noqa: E402
    AppendOnlyJournal,
    HarnessConfig,
    ModelConfig,
    OpenAICompatibleClient,
    SQLTaskAdapter,
    run_task,
)
from pilot_sql.tasks import dataset_hash, load_tasks  # noqa: E402
from pilot_sql.verifier import verify  # noqa: E402

PATTERNS = ("direct", "self_refine", "verifier_loop", "memory_assisted")
MAIN_SEED = 20261008
REPEAT_SEEDS = (20261009, 20261010)
MODEL = os.environ.get("PILOT_MODEL", "ministral-3-8b-instruct-2512")
BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234")
MODEL_HASH = os.environ.get("PILOT_MODEL_ARTIFACT_HASH", "")
PROMPTS = {
    "direct": "Write one correct SQLite query. Return only SQL, without explanation.",
    "self_refine": "Write one correct SQLite query. Return only SQL, without explanation.",
    "verifier_loop": "Write one correct SQLite query. Return only SQL, without explanation.",
    "memory_assisted": "Write one correct SQLite query. Use the public examples only as structural hints. Return only SQL, without explanation.",
}


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def source_hash() -> str:
    paths = [
        Path(__file__).absolute(),
        ROOT / "pilot_common" / "runner.py",
        ROOT / "pilot_common" / "adapters.py",
        ROOT / "pilot_sql" / "tasks.py",
        ROOT / "pilot_sql" / "verifier.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    return json.loads((ROOT / "pilot_sql" / "splits_manifest.json").read_text(encoding="utf-8"))


def select_tasks() -> tuple[list[tuple[str, dict]], list[dict], dict]:
    manifest = load_manifest()
    by_id = {task["task_id"]: task for task in load_tasks()}
    wanted = list(manifest["first_experiment_ids"])
    heldout = set(manifest["splits"]["held-out"])
    if len(wanted) != 40 or len(set(wanted)) != 40:
        raise RuntimeError("first_experiment_ids must contain exactly 40 unique identifiers")
    overlap = sorted(set(wanted) & heldout)
    if overlap:
        raise RuntimeError(f"first_experiment_ids intersects held-out: {overlap}")
    missing = sorted(set(wanted) - set(by_id))
    if missing:
        raise RuntimeError(f"unknown first_experiment_ids: {missing}")
    split_by_id = {
        task_id: ("heldout" if split == "held-out" else split)
        for split, ids in manifest["splits"].items()
        for task_id in ids
    }
    selected = [(split_by_id[task_id], by_id[task_id]) for task_id in wanted]
    # Static memory is restricted to the permitted search split.  Its entries
    # contain public fields only; no answer SQL is ever stored or retrieved.
    memory = [by_id[task_id] for task_id in manifest["splits"]["search"]]
    return selected, memory, manifest


def task_features(task: dict) -> dict:
    features = dict(task.get("features", {}))
    features.update({
        "n_words": len(task["description"].split()),
        "n_chars": len(task["description"]),
        "n_allowed_operations": len(task.get("allowed_operations", [])),
    })
    return features


class PublicMemoryBank:
    """Deterministic lexical retrieval over search examples with no answers."""

    def __init__(self, items: list[dict], k: int = 2):
        self.items = list(items)
        self.k = k
        self.calls = 0
        self.last_examples: list[dict] = []

    def retrieve(self, description: str) -> list[dict]:
        self.calls += 1
        query = set(description.lower().split())
        scored = []
        for item in self.items:
            if item["description"] == description:  # Exclude the current task.
                continue
            words = set(item["description"].lower().split())
            score = len(query & words) / (len(query | words) or 1)
            scored.append((score, item["task_id"], item))
        scored.sort(key=lambda row: (-row[0], row[1]))
        adapter = SQLTaskAdapter()
        self.last_examples = [adapter.public_task(item) for _, _, item in scored[: self.k]]
        return self.last_examples


def make_config(pattern: str, seed: int, hsource: str) -> HarnessConfig:
    model = ModelConfig(
        backend="openai_compatible",
        model=MODEL,
        base_url=BASE_URL,
        temperature=0.0,
        seed=seed,
        max_tokens=512,
        num_ctx=4096,
        timeout_s=1800.0,
    )
    return HarnessConfig(
        system_prompt=PROMPTS[pattern],
        pattern=pattern,
        max_repairs=1 if pattern == "verifier_loop" else 0,
        memory_k=2 if pattern == "memory_assisted" else 0,
        use_visible_verifier=True,
        stop_rule="after_success_or_one_repair" if pattern == "verifier_loop" else "after_final_verifier",
        tool_policy={"verifier": "pilot_sql.verifier.verify", "memory": "public_search_only"},
        model=model,
        seed=seed,
        harness_source_hash=hsource,
    )


def _final_verifier(steps: list[dict]) -> dict:
    matches = [step.get("output") for step in steps if step.get("kind") == "verifier"]
    return matches[-1] if matches and isinstance(matches[-1], dict) else {}


def public_record(task: dict, split: str, seed: int, pattern: str, trajectory, cfg: HarnessConfig) -> dict:
    data = trajectory.to_dict()
    steps = data["steps"]
    if pattern == "memory_assisted":
        # pilot_common performs retrieval before its first recorded model step.
        # Add the call to the public trajectory without recording any private data.
        steps = [{
            "index": 0,
            "kind": "tool",
            "name": "memory_retrieval",
            "input": {"description": task["description"]},
            "output": {"source_split": "search", "k": cfg.memory_k},
            "messages": [],
            "tool_result": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "calls": 1,
            "latency_s": 0.0,
            "error_type": None,
            "error": None,
        }] + [{**step, "index": int(step["index"]) + 1} for step in steps]
    verification = _final_verifier(steps)
    model_id = f"lmstudio/{MODEL}" + (f"@sha256:{MODEL_HASH}" if MODEL_HASH else "")
    return {
        "task_id": task["task_id"],
        "task_template": task["template_group"],
        "split": split,
        "seed": seed,
        "harness": pattern,
        "model": model_id,
        "success": int(bool(trajectory.success)),
        "tokens": trajectory.total_prompt_tokens + trajectory.total_completion_tokens,
        "model_calls": trajectory.model_calls,
        "tool_calls": trajectory.tool_calls + (1 if pattern == "memory_assisted" else 0),
        "latency_s": trajectory.latency_s,
        "task_features": task_features(task),
        "trajectory": steps,
        "config_hash": cfg.config_hash,
        "source_hash": cfg.harness_source_hash,
        "error_type": verification.get("error_code"),
        "infrastructure_error": None,
    }


def leakage_errors(record: dict, reference_queries: list[str]) -> list[str]:
    errors = []
    public_inputs = {
        "task_id": record["task_id"],
        "task_template": record["task_template"],
        "task_features": record["task_features"],
    }
    chunks = [json.dumps(public_inputs, ensure_ascii=False)]
    for step in record["trajectory"]:
        if step.get("kind") == "model":
            chunks.append(str(step.get("input", "")))
            chunks.append(json.dumps(step.get("messages", []), ensure_ascii=False))
        elif step.get("name") == "memory_retrieval":
            chunks.append(json.dumps(step.get("input"), ensure_ascii=False))
            chunks.append(json.dumps(step.get("output"), ensure_ascii=False))
    haystack = "\n".join(chunks)
    lowered = haystack.lower()
    for marker in ("reference_sql", "hidden_secret"):
        if marker in lowered:
            errors.append(f"forbidden marker in public input: {marker}")
    for query in reference_queries:
        if query and query in haystack:
            errors.append("exact reference query in public input")
            break
    return errors


def load_outcomes(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def completed_keys(path: Path) -> set[tuple[str, str, int, str]]:
    return {
        (str(row["task_id"]), str(row["harness"]), int(row["seed"]), str(row["config_hash"]))
        for row in load_outcomes(path)
        if not row.get("infrastructure_error")
    }


def _write_new_or_validate(path: Path, value: Any) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"refusing to overwrite changed frozen artifact: {path}")
        return
    path.write_text(rendered, encoding="utf-8")


@contextmanager
def collector_lock(out_root: Path):
    lock = out_root / ".lmstudio_collector.lock"
    acquired = False
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "collector": "sql_signal"}))
        acquired = True
        yield
    finally:
        if acquired and lock.exists():
            lock.unlink()


def check_static_smoke(adapter: SQLTaskAdapter, sample_task: dict) -> list[str]:
    errors = []
    fenced = adapter.extract_output("```sql\nSELECT 1;\n```")
    plain = adapter.extract_output("SELECT 1;")
    if fenced != "SELECT 1" or plain != "SELECT 1":
        errors.append("SQL extraction failed for fenced or plain-text response")
    unsafe = verify(sample_task, "DELETE FROM customers")
    multiple = verify(sample_task, "SELECT 1; SELECT 2")
    if unsafe.get("error_code") != "unsafe_query":
        errors.append("unsafe query was not blocked")
    if multiple.get("error_code") != "multiple_statements":
        errors.append("multiple statements were not blocked")
    return errors


def check_smoke(rows: list[dict], reference_queries: list[str]) -> list[str]:
    errors = []
    for row in rows:
        models = [s for s in row["trajectory"] if s.get("kind") == "model"]
        verifiers = [s for s in row["trajectory"] if s.get("kind") == "verifier"]
        retrievals = [s for s in row["trajectory"] if s.get("name") == "memory_retrieval"]
        expected_models = {"direct": (1,), "self_refine": (2,), "verifier_loop": (1, 2), "memory_assisted": (1,)}[row["harness"]]
        if len(models) not in expected_models:
            errors.append(f"{row['task_id']}/{row['harness']}: model calls={len(models)}")
        if row["harness"] == "verifier_loop":
            if len(verifiers) != len(models):
                errors.append(f"{row['task_id']}/verifier_loop: verifier/model calls differ")
        elif len(verifiers) != 1:
            errors.append(f"{row['task_id']}/{row['harness']}: verifier calls={len(verifiers)}")
        expected_retrievals = 1 if row["harness"] == "memory_assisted" else 0
        if len(retrievals) != expected_retrievals:
            errors.append(f"{row['task_id']}/{row['harness']}: retrieval calls={len(retrievals)}")
        errors.extend(f"{row['task_id']}/{row['harness']}: {msg}" for msg in leakage_errors(row, reference_queries))
    return errors


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20261008")
    parser.add_argument("--out-root", default=str(ROOT / "runs"))
    parser.add_argument("--skip-main", action="store_true")
    args = parser.parse_args(argv)

    requested_backend = os.environ.get("PILOT_BACKEND", "lmstudio").lower()
    if requested_backend != "lmstudio":
        raise SystemExit(f"this collector requires PILOT_BACKEND=lmstudio, got {requested_backend!r}")

    selected, memory_items, split_manifest = select_tasks()
    all_tasks = load_tasks()
    reference_queries = [task["reference_sql"] for task in all_tasks]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"sql_signal_{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = run_dir / "outcomes.jsonl"
    trajectories = AppendOnlyJournal(run_dir / "trajectories.jsonl")
    infra_journal = AppendOnlyJournal(run_dir / "infrastructure_errors.jsonl")
    outcomes_path.touch(exist_ok=True)
    hsource = source_hash()

    task_manifest = {
        "dataset": "pilot_sql",
        "dataset_hash": dataset_hash(),
        "split_manifest_hash": canonical_hash(split_manifest),
        "selection_rule": "exact ordered first_experiment_ids",
        "tasks": [{"task_id": task["task_id"], "split": split, "template_group": task["template_group"]} for split, task in selected],
        "task_ids": [task["task_id"] for _, task in selected],
        "task_ids_sha256": canonical_hash([task["task_id"] for _, task in selected]),
        "memory_task_ids": [task["task_id"] for task in memory_items],
        "memory_fields": ["task_id", "description", "schema", "allowed_operations", "features", "template_group"],
        "heldout_included": False,
        "heldout_overlap": [],
    }
    configs = {
        f"{pattern}:{seed}": make_config(pattern, seed, hsource).config_hash
        for pattern in PATTERNS
        for seed in (MAIN_SEED, *REPEAT_SEEDS)
    }
    base_config = {
        "backend": "lmstudio",
        "endpoint": os.environ.get("LM_STUDIO_CHAT_URL", BASE_URL.rstrip("/") + "/v1/chat/completions"),
        "model": MODEL,
        "model_artifact_hash": MODEL_HASH or None,
        "temperature": 0.0,
        "max_tokens": 512,
        "num_ctx": 4096,
        "main_seed": MAIN_SEED,
        "repeat_seeds": list(REPEAT_SEEDS),
        "patterns": list(PATTERNS),
        "prompts": PROMPTS,
        "source_hash": hsource,
        "config_hashes": configs,
        "task_manifest_hash": canonical_hash(task_manifest),
        "command": f"python pilot/sql_signal_collect.py --run-id {args.run_id}",
        "platform": platform.platform(),
        "python": sys.version,
    }
    _write_new_or_validate(run_dir / "task_manifest.json", task_manifest)
    _write_new_or_validate(run_dir / "config.json", base_config)

    memory = PublicMemoryBank(memory_items)
    client_by_seed: dict[int, OpenAICompatibleClient] = {}

    def run_one(split: str, task: dict, pattern: str, seed: int):
        cfg = make_config(pattern, seed, hsource)
        key = (task["task_id"], pattern, seed, cfg.config_hash)
        if key in completed_keys(outcomes_path):
            return "skip", next(row for row in load_outcomes(outcomes_path) if (
                row["task_id"], row["harness"], int(row["seed"]), row["config_hash"]
            ) == key)
        try:
            client = client_by_seed.setdefault(seed, OpenAICompatibleClient(cfg.model))
            trajectory = run_task(
                task,
                cfg,
                client,
                lambda answer, current=task: verify(current, answer),
                memory=memory,
                adapter=SQLTaskAdapter(),
            )
            if trajectory.error_type:
                raise RuntimeError(f"model infrastructure error: {trajectory.error_type}")
            record = public_record(task, split, seed, pattern, trajectory, cfg)
            leaks = leakage_errors(record, reference_queries)
            if leaks:
                raise RuntimeError("leakage guard failed: " + "; ".join(leaks))
        except Exception as exc:
            infra = {
                "record_type": "infrastructure_error",
                "task_id": task["task_id"],
                "split": split,
                "harness": pattern,
                "seed": seed,
                "config_hash": cfg.config_hash,
                "source_hash": hsource,
                "infrastructure_error": f"{type(exc).__name__}: {exc}",
            }
            infra_journal.append(infra)
            return "infra", infra
        with outcomes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        trajectories.append({"record_type": "trajectory", **record})
        return "done", record

    with collector_lock(out_root):
        smoke_rows = []
        for split, task in selected[:2]:
            for pattern in PATTERNS:
                status, row = run_one(split, task, pattern, MAIN_SEED)
                if status == "infra":
                    raise SystemExit("smoke infrastructure failure; main collection was not started")
                smoke_rows.append(row)
        smoke_errors = check_static_smoke(SQLTaskAdapter(), selected[0][1])
        smoke_errors.extend(check_smoke(smoke_rows, reference_queries))
        before = len(load_outcomes(outcomes_path))
        duplicate_status, _ = run_one(selected[0][0], selected[0][1], PATTERNS[0], MAIN_SEED)
        after = len(load_outcomes(outcomes_path))
        if duplicate_status != "skip" or before != after:
            smoke_errors.append("repeat execution created a duplicate outcome")
        smoke_lines = [
            "# SQL signal collector smoke report",
            "",
            f"Run: `sql_signal_{args.run_id}`",
            f"Rows checked: {len(smoke_rows)} (2 tasks × 4 patterns)",
            f"Errors: {len(smoke_errors)}",
            "",
        ]
        smoke_lines += [f"- {error}" for error in smoke_errors] or [
            "- PASS: model/verifier/retrieval call counts",
            "- PASS: no private markers or exact reference SQL in public inputs",
            "- PASS: fenced and plain-text SQL extraction",
            "- PASS: unsafe and multiple-statement queries blocked",
            "- PASS: resume did not create a duplicate",
        ]
        smoke_path = run_dir / "smoke_report.md"
        rendered_smoke = "\n".join(smoke_lines) + "\n"
        if smoke_path.exists() and smoke_path.read_text(encoding="utf-8") != rendered_smoke:
            raise RuntimeError(f"refusing to overwrite changed smoke report: {smoke_path}")
        smoke_path.write_text(rendered_smoke, encoding="utf-8")
        if smoke_errors:
            raise SystemExit("smoke failed; main collection was not started: " + "; ".join(smoke_errors))
        if args.skip_main:
            return

        repeat_ids = {task["task_id"] for _, task in selected[:20]}
        jobs = [
            (split, task, pattern, seed)
            for split, task in selected
            for pattern in PATTERNS
            for seed in ((MAIN_SEED, *REPEAT_SEEDS) if task["task_id"] in repeat_ids else (MAIN_SEED,))
        ]
        for index, (split, task, pattern, seed) in enumerate(jobs, 1):
            status, row = run_one(split, task, pattern, seed)
            if status == "infra":
                raise SystemExit(f"infrastructure failure at job {index}/{len(jobs)}: {row['infrastructure_error']}")
            if index % 10 == 0 or index == len(jobs):
                print(f"[{index}/{len(jobs)}] {status}", flush=True)

    rows = load_outcomes(outcomes_path)
    expected = {
        (task["task_id"], pattern, seed, make_config(pattern, seed, hsource).config_hash)
        for _, task in selected
        for pattern in PATTERNS
        for seed in ((MAIN_SEED, *REPEAT_SEEDS) if task["task_id"] in repeat_ids else (MAIN_SEED,))
    }
    actual = completed_keys(outcomes_path)
    duplicates = len(rows) - len(actual)
    error_counts = Counter(row.get("error_type") or "success" for row in rows)
    integrity = {
        "expected_pairs": len(expected),
        "completed_pairs": len(actual & expected),
        "missing_pairs": [list(key) for key in sorted(expected - actual)],
        "unexpected_pairs": [list(key) for key in sorted(actual - expected)],
        "duplicate_rows": duplicates,
        "error_type_counts": dict(sorted(error_counts.items())),
        "leakage_errors": sum((leakage_errors(row, reference_queries) for row in rows), []),
        "smoke_errors": smoke_errors,
        "heldout_evaluated": False,
        "repeat_tasks": sorted(repeat_ids),
        "determinism_note": "At temperature=0, repeated seeds may be identical and are not interpreted as independent stochastic measurements.",
    }
    _write_new_or_validate(run_dir / "integrity_report.json", integrity)
    readme = (
        "# SQL signal collector\n\n"
        "Harness — внешняя схема работы агента. Основной запуск и автоматическое продолжение:\n\n"
        "```powershell\n"
        "$env:PILOT_BACKEND='lmstudio'\n"
        f"$env:PILOT_MODEL='{MODEL}'\n"
        f"python pilot/sql_signal_collect.py --run-id {args.run_id}\n"
        "```\n\n"
        "Только smoke test (проверка на малой выборке): добавьте `--skip-main`. "
        "Продолжение автоматически пропускает ключ `(task_id, pattern, seed, config_hash)`. "
        "При изменении замороженной конфигурации или исходников запуск отказывается смешивать результаты. "
        "Held-out и обучение policy (стратегии выбора) не запускались.\n"
    )
    readme_path = run_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
