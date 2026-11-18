"""Resumable, cached evaluation of a harness (component map) on a benchmark split.

    python -m harness_lab.evaluate --bench mbpp --split test --harness seed --name seed_test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from harness_lab._paths import RESULTS
from harness_lab.benchmarks import load_humaneval, load_humaneval_split, load_mbpp
from harness_lab.code_harness import SEED_COMPONENTS, run_task
from harness_lab.llm import LLMSettings, OllamaClient


def components_hash(components: dict[str, str], extra: dict | None = None) -> str:
    blob = json.dumps({"c": components, "x": extra or {}}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_tasks(bench: str, split: str) -> list[dict]:
    if bench == "mbpp":
        return load_mbpp(split)
    if bench == "humaneval":
        return load_humaneval_split(split) if split in ("train", "val", "test") else load_humaneval()
    raise ValueError(f"unknown code benchmark: {bench}")


def evaluate(components: dict[str, str], tasks: list[dict], settings: LLMSettings, *,
             use_repair: bool = True, cache: Path | None = None, tag: str = "", solver=None) -> list[dict]:
    """Return one row per task; rows are cached by (harness hash, model, settings, task)."""
    key_extra = {"repair": use_repair, "model": settings.model, "temp": settings.temperature,
                 "seed": settings.seed, "ctx": settings.num_ctx, "max": settings.max_tokens, "tag": tag}
    chash = components_hash(components, key_extra)
    store: dict[str, dict] = {}
    if cache and cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            store[f"{row['_h']}:{row['task_id']}"] = row
    client = OllamaClient(settings)
    rows = []
    for task in tasks:
        k = f"{chash}:{task['task_id']}"
        if k in store:
            rows.append(store[k])
            continue
        row = (solver or run_task)(task, components, client, use_repair=use_repair)
        row["_h"] = chash
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "pass_at_1": sum(r["success"] for r in rows) / n if n else None,
        "tokens_per_task": sum(r["tokens"] for r in rows) / n if n else None,
        "repair_rate": sum(r["repairs"] > 0 for r in rows) / n if n else None,
        "latency_s_per_task": sum(r["latency_s"] for r in rows) / n if n else None,
    }


def resolve_harness(spec: str) -> dict[str, str]:
    if spec == "seed":
        return dict(SEED_COMPONENTS)
    data = json.loads(Path(spec).read_text(encoding="utf-8"))
    return data.get("components", data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="mbpp")
    ap.add_argument("--split", default="test")
    ap.add_argument("--harness", default="seed", help="'seed' or path to JSON component map")
    ap.add_argument("--no-repair", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--base-url", default=LLMSettings.base_url,
                    help="Ollama URL, or an OpenAI-compatible URL ending in /v1 (LM Studio)")
    ap.add_argument("--num-ctx", type=int, default=LLMSettings.num_ctx)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    settings = LLMSettings(model=args.model, base_url=args.base_url, num_ctx=args.num_ctx)
    tasks = load_tasks(args.bench, args.split)[: args.limit]
    started = time.time()
    rows = evaluate(resolve_harness(args.harness), tasks, settings, use_repair=not args.no_repair,
                    cache=RESULTS / "cache" / f"{args.bench}.jsonl")
    out = RESULTS / "evals" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {**summarize(rows), "bench": args.bench, "split": args.split, "model": args.model,
               "harness": args.harness, "repair": not args.no_repair, "wall_s": time.time() - started}
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
