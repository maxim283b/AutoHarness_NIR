"""Paired comparison of two test-set evaluations (results.csv from run_eval.py).

  python analyze.py --a runs/std_test --b runs/con_test --out runs/compare.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest


def load(run):
    with open(Path(run) / "results.csv", encoding="utf-8") as f:
        return {int(r["task_id"]): r for r in csv.DictReader(f)}


def compare(a_dir, b_dir, n_boot=10_000, seed=0):
    A, B = load(a_dir), load(b_dir)
    ids = sorted(set(A) & set(B))
    a = np.array([int(A[i]["pass"]) for i in ids])
    b = np.array([int(B[i]["pass"]) for i in ids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    diffs = b[idx].mean(1) - a[idx].mean(1)
    b_only, a_only = int(((b == 1) & (a == 0)).sum()), int(((a == 1) & (b == 0)).sum())
    disc = a_only + b_only
    p_mcnemar = binomtest(b_only, disc, 0.5).pvalue if disc else 1.0

    def tok(R):
        return float(np.mean([int(R[i]["prompt_tokens"]) + int(R[i]["completion_tokens"]) for i in ids]))

    def post(x):
        k, n = int(x.sum()), len(x)
        lo, hi = beta.ppf([0.025, 0.975], k + 1, n - k + 1)
        return {"k": k, "n": n, "mean": (k + 1) / (n + 2), "ci95": [float(lo), float(hi)]}

    return {
        "a": str(a_dir), "b": str(b_dir), "n": len(ids),
        "pass_a": float(a.mean()), "pass_b": float(b.mean()), "diff_b_minus_a": float(b.mean() - a.mean()),
        "diff_ci95_paired_bootstrap": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "b_only": b_only, "a_only": a_only, "mcnemar_exact_p": float(p_mcnemar),
        "beta_posterior_a": post(a), "beta_posterior_b": post(b),
        "tokens_per_task_a": tok(A), "tokens_per_task_b": tok(B),
        "repair_rate_a": float(np.mean([int(A[i]["repair_used"]) for i in ids])),
        "repair_rate_b": float(np.mean([int(B[i]["repair_used"]) for i in ids])),
        "flipped_to_pass": [i for i in ids if B[i]["pass"] == "1" and A[i]["pass"] == "0"],
        "flipped_to_fail": [i for i in ids if A[i]["pass"] == "1" and B[i]["pass"] == "0"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default=None)
    x = ap.parse_args()
    r = compare(x.a, x.b)
    s = json.dumps(r, indent=1)
    if x.out:
        Path(x.out).write_text(s, encoding="utf-8")
    print(s)
