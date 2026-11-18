"""Paired final analysis with exact provenance checks; no optimization access."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import random

from run import OUT, SPEC, verify, result_valid, sha, save


def exact_mcnemar(a, b):
    better = sum(x == 0 and y == 1 for x, y in zip(a, b))
    worse = sum(x == 1 and y == 0 for x, y in zip(a, b))
    discordant = better + worse
    if not discordant:
        return {"better": better, "worse": worse, "p_two_sided": 1.0}
    tail = sum(math.comb(discordant, k) for k in range(min(better, worse) + 1)) / 2 ** discordant
    return {"better": better, "worse": worse, "p_two_sided": min(1.0, 2 * tail)}


def bootstrap(a, b, seed=20261112, draws=20000):
    rng = random.Random(seed)
    n = len(a)
    values = []
    for _ in range(draws):
        indices = [rng.randrange(n) for _ in range(n)]
        values.append(100 * sum(b[i] - a[i] for i in indices) / n)
    values.sort()
    return [values[int(.025 * draws)], values[min(draws - 1, int(.975 * draws))]]


def analyze(allow_first_16=False):
    verify()
    plan = json.loads((OUT / "final/plan.json").read_text())
    manifest = json.loads((SPEC / "manifest.json").read_text())
    if plan["manifest_sha256"] != sha(SPEC / "manifest.json") or set(plan["test"]) != set(manifest["test"]):
        raise RuntimeError("Final plan/manifest mismatch")
    tasks = plan["test"][:16] if allow_first_16 else plan["test"]
    if allow_first_16 and tasks != manifest["test_fallback_first_16"]:
        raise RuntimeError("Fallback subset mismatch")
    index = {(task, label): (i, first) for i, (task, label, first) in enumerate(plan["schedule"])}
    rows = []
    for task in tasks:
        for label in ("H0", "H1", "H2"):
            path = OUT / "final/results" / label / hashlib.sha256(task.encode()).hexdigest()[:16] / "result.json"
            if not path.exists():
                raise RuntimeError(f"Missing final result: {label} {task}")
            result = json.loads(path.read_text())
            attempt = json.loads((path.parent / "attempt.json").read_text())
            certificate_path = SPEC.parent / "results/autosaddler_reproduction/fullsource_image_preflight" / (hashlib.sha256(task.encode()).hexdigest()[:20] + ".json")
            certificate = json.loads(certificate_path.read_text())
            if certificate.get("task_id") != task or certificate.get("passed") is not True:
                raise RuntimeError("Task image certificate mismatch")
            role = "target_final_baseline" if label == "H0" else "target_final_optimized"
            execution_index, first = index[(task, label)]
            expected = {"run_id": plan["run_id"], "label": label, "task_id": task,
                        "benchmark": "swe_bench_pro", "candidate_id": plan["candidate_ids"][label],
                        "model": "opencode-go/qwen3.7-plus", "role": role,
                        "execution_index": execution_index, "planned_first_label": first,
                        "candidate_hash": attempt.get("candidate_hash"),
                        "grading_image_id": certificate["image_id"]}
            if not result_valid(result, expected, role) or any(attempt.get(k) != v for k, v in expected.items()):
                raise RuntimeError(f"Invalid official result: {path}")
            rows.append(result)
    scores = {label: [int(next(row["score"] for row in rows if row["task_id"] == task and row["label"] == label))
                      for task in tasks] for label in ("H0", "H1", "H2")}
    pairs = {}
    for a, b in (("H0", "H1"), ("H0", "H2"), ("H1", "H2")):
        pairs[f"{b}_minus_{a}"] = {"delta_pp": 100 * (sum(scores[b]) - sum(scores[a])) / len(tasks),
                                    "bootstrap_95_pp": bootstrap(scores[a], scores[b]),
                                    "mcnemar": exact_mcnemar(scores[a], scores[b])}
    totals = {}
    for label in scores:
        role = "target_final_baseline" if label == "H0" else "target_final_optimized"
        buckets = [row["usage"][role] for row in rows if row["label"] == label]
        totals[label] = {"pass_at_1": sum(scores[label]) / len(tasks), "passed": sum(scores[label]),
                         "calls": sum(bucket["calls"] for bucket in buckets),
                         "input_tokens": sum(bucket["input_tokens"] for bucket in buckets),
                         "output_tokens": sum(bucket["output_tokens"] for bucket in buckets),
                         "total_tokens": sum(bucket["total_tokens"] for bucket in buckets)}
    report = {"benchmark": "SWE-bench Pro V1", "tasks": len(tasks), "mode": "predeclared_first_16" if allow_first_16 else "full_40",
              "prior_exposure": manifest["prior_observations_disclosure"],
              "candidate_ids": plan["candidate_ids"], "results": totals, "paired": pairs,
              "note": "Exact paired outcomes; bootstrap intervals describe this sampled task set. Earlier design observations may affect external validity."}
    output = OUT / ("analysis-first16.json" if allow_first_16 else "analysis-full40.json")
    save(output, report)
    with (OUT / ("paired-first16.csv" if allow_first_16 else "paired-full40.csv")).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task_id", "H0", "H1", "H2"))
        for i, task in enumerate(tasks):
            writer.writerow((task, scores["H0"][i], scores["H1"][i], scores["H2"][i]))
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-16", action="store_true")
    args = parser.parse_args()
    print(json.dumps(analyze(args.first_16), ensure_ascii=False, indent=2))
