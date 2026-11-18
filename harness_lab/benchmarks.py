"""Benchmark loaders producing unified task dicts:
{task_id, prompt, imports, tests, visible_test}."""
from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

from harness_lab._paths import RESULTS, ROOT

MBPP_DIR = ROOT / "pilot_baseline" / "data"
HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"


def load_mbpp(split: str) -> list[dict]:
    """split in train (30) / val (20) / test (50) from the repo's frozen manifest."""
    tasks = []
    for line in (MBPP_DIR / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        tasks.append({
            "task_id": f"mbpp-{r['task_id']}",
            "prompt": r["prompt"],
            "imports": r["test_imports"],
            "tests": r["test_list"],
            "visible_test": r["test_list"][0],
        })
    return tasks


def load_humaneval() -> list[dict]:
    """HumanEval (164). No visible test: the harness is single-shot there (no repair signal)."""
    cache = RESULTS / "data" / "HumanEval.jsonl.gz"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HUMANEVAL_URL, cache)
    tasks = []
    with gzip.open(cache, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            program = f"{r['test']}\ncheck({r['entry_point']})"
            tasks.append({
                "task_id": r["task_id"].replace("/", "-").lower(),
                "prompt": ("Complete the following Python function. Return the full function "
                           f"(signature included) in one code block.\n\n{r['prompt']}"),
                "imports": [],
                "tests": [program],
                "visible_test": None,
            })
    return tasks


def load_sql(split_key: str = "search") -> list[dict]:
    from pilot_sql.tasks import load_tasks
    return [t for t in load_tasks() if t.get("split") == split_key]


HE_SPLITS = {"train": (0, 40), "val": (40, 64), "test": (64, 164)}


def load_humaneval_split(split: str) -> list[dict]:
    """Frozen shuffled 40/24/100 split of HumanEval (seed 12345); test never used for optimization."""
    import random
    tasks = sorted(load_humaneval(), key=lambda t: t["task_id"])
    random.Random(12345).shuffle(tasks)
    lo, hi = HE_SPLITS[split]
    return tasks[lo:hi]
