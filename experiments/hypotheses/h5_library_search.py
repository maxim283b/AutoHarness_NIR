"""H5: a no-LLM-proposer search over a small hand-curated instruction library, selected on the
SAME small train+dev budget, matches AutoSaddler-style optimization at ~0 proposer tokens.

Library = alternative solver_prompt/checklist/repair_prompt texts (generic best practices, no
task-specific leakage).  Selection: best mean on (train n + dev n) tasks, ties -> seed.
Then evaluated on MBPP-test against the seed.
"""
from __future__ import annotations

import json
import random

from harness_lab._paths import RESULTS
from harness_lab.benchmarks import load_mbpp
from harness_lab.code_harness import SEED_COMPONENTS
from harness_lab.evaluate import evaluate, summarize
from harness_lab.llm import LLMSettings
from harness_lab.stats import paired_compare

LIBRARY = {
    "seed": {},
    "think_edge_cases": {"checklist": "Use the exact function name from the test. Do not print or read input. "
                                      "Before writing, consider edge cases (empty input, negatives, duplicates)."},
    "match_test_format": {"checklist": "Use the exact function name and argument order from the test. Return "
                                       "values of exactly the type shown in the test (tuple vs list, int vs float)."},
    "minimal": {"solver_prompt": "Write the simplest correct Python function for the task. Return only one "
                                 "```python``` code block."},
    "stdlib_only": {"checklist": "Use the exact function name from the test. Use only the Python standard library "
                                 "and include required imports inside the code block."},
    "repair_diff": {"repair_prompt": "The previous solution failed the test. Compare the expected value in the test "
                                     "with what the code returns, find the logic error, and return only one "
                                     "corrected ```python``` code block."},
}


def run(seed: int = 0, n_train: int = 12, n_dev: int = 10) -> dict:
    rng = random.Random(seed)
    train, dev = load_mbpp("train"), load_mbpp("val")
    rng.shuffle(train), rng.shuffle(dev)
    search = train[:n_train] + dev[:n_dev]
    settings, cache = LLMSettings(), RESULTS / "cache" / "mbpp.jsonl"
    scores = {}
    for name, patch in LIBRARY.items():
        rows = evaluate({**SEED_COMPONENTS, **patch}, search, settings, cache=cache)
        scores[name] = sum(r["reward"] for r in rows) / len(rows)
    best = max(LIBRARY, key=lambda k: (scores[k], k == "seed"))
    test = load_mbpp("test")
    seed_rows = evaluate(dict(SEED_COMPONENTS), test, settings, cache=cache)
    best_rows = evaluate({**SEED_COMPONENTS, **LIBRARY[best]}, test, settings, cache=cache)
    return {"data_seed": seed, "search_n": len(search), "search_scores": scores, "selected": best,
            "proposer_tokens": 0, "test_selected": summarize(best_rows),
            "test_paired_vs_seed": paired_compare([r["reward"] for r in seed_rows], [r["reward"] for r in best_rows])}


def main() -> None:
    outs = [run(seed) for seed in (0, 1, 2)]
    path = RESULTS.parent / "hypotheses" / "h5_library_search.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(outs, indent=1), encoding="utf-8")
    for o in outs:
        print(o["data_seed"], o["selected"], round(o["test_paired_vs_seed"]["delta"], 3), o["test_paired_vs_seed"]["p_mcnemar"])


if __name__ == "__main__":
    main()
