"""H4: semantically equivalent prompt paraphrases move pass@1 by more than the typical
gain AutoSaddler reports on small data, so tiny dev/train deltas are not evidence.

Evaluates the seed and K paraphrases of the SAME instructions on MBPP-test (50) and reports
the spread plus how often a paraphrase 'significantly' differs from the seed (false discovery).
"""
from __future__ import annotations

import json
import statistics

from harness_lab._paths import RESULTS
from harness_lab.benchmarks import load_mbpp
from harness_lab.code_harness import SEED_COMPONENTS
from harness_lab.evaluate import evaluate
from harness_lab.llm import LLMSettings
from harness_lab.stats import paired_compare

PARAPHRASES = {
    "p1": ("You are an expert Python developer. Implement a correct, self-contained Python function for "
           "the task below. Reply with a single ```python``` code block only."),
    "p2": ("Solve the programming task with a complete, standalone Python function. Output exactly one "
           "```python``` block and nothing else."),
    "p3": ("As a careful Python programmer, write a function that solves the task and satisfies the given "
           "test. Respond with only one ```python``` fenced code block."),
    "p4": ("Write Python code for the following task. The code must be self-contained and correct. Return "
           "one ```python``` code block, no explanation."),
    "p5": ("Your job: produce a working Python function for the request. Format: a single ```python``` "
           "code block, without any prose."),
}


def main() -> None:
    settings = LLMSettings()
    tasks = load_mbpp("test")
    cache = RESULTS / "cache" / "mbpp.jsonl"
    base = evaluate(dict(SEED_COMPONENTS), tasks, settings, cache=cache)
    base_scores = [r["reward"] for r in base]
    out = {"seed_pass": sum(base_scores) / len(base_scores), "variants": {}}
    for name, text in PARAPHRASES.items():
        comps = {**SEED_COMPONENTS, "solver_prompt": text}
        rows = evaluate(comps, tasks, settings, cache=cache)
        out["variants"][name] = paired_compare(base_scores, [r["reward"] for r in rows])
    passes = [v["mean_b"] for v in out["variants"].values()] + [out["seed_pass"]]
    out["pass_min"], out["pass_max"] = min(passes), max(passes)
    out["pass_sd"] = statistics.pstdev(passes)
    out["max_abs_delta"] = max(abs(v["delta"]) for v in out["variants"].values())
    out["n_significant_p05"] = sum(v["p_mcnemar"] < 0.05 for v in out["variants"].values())
    path = RESULTS.parent / "hypotheses" / "h4_noise_floor.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "variants"}, indent=1))


if __name__ == "__main__":
    main()
