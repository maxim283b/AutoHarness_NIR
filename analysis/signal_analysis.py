"""Pre-registered signal analysis for standardized harness JSONL.

The module deliberately treats heldout rows as scoring-only data.  Model selection,
thresholds and reward coefficients are constants from docs/pre_registered_analysis.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LAMBDAS = [("main", .10, .02), ("low_cost", .05, .01),
           ("high_cost", .20, .05), ("quality_first", 0., 0.)]
BOOTSTRAPS = 10_000
PRACTICAL_REWARD = .05
PRACTICAL_SUCCESS = .02


def reward(row: dict, lt: float = .10, ltool: float = .02) -> float:
    return float(row["success"]) - lt * float(row["tokens"]) / 1000 - ltool * float(row["tool_calls"])


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{no}: invalid JSON: {exc}") from exc
        required = {"task_id", "task_template", "split", "seed", "harness", "model",
                    "success", "tokens", "tool_calls", "latency_s", "task_features", "trajectory"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path}:{no}: missing fields {sorted(missing)}")
        if row["split"] not in {"search", "validation", "online", "heldout"}:
            raise ValueError(f"{path}:{no}: invalid split {row['split']!r}")
        if row["success"] not in (0, 1, False, True):
            raise ValueError(f"{path}:{no}: success must be binary")
        if not isinstance(row["trajectory"], list):
            raise ValueError(f"{path}:{no}: trajectory must be a list")
        for k in ("tokens", "tool_calls", "latency_s"):
            if not isinstance(row[k], (int, float)) or row[k] < 0:
                raise ValueError(f"{path}:{no}: {k} must be non-negative numeric")
        rows.append(row)
    if not rows:
        raise ValueError("input JSONL is empty")
    return rows


def ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    m = statistics.fmean(values)
    if len(values) < 2:
        return m, m, m
    se = statistics.stdev(values) / math.sqrt(len(values))
    return m, m - 1.96 * se, m + 1.96 * se


def bootstrap(values: list[float], seed: int, n: int = BOOTSTRAPS) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n)]
    means.sort()
    return statistics.fmean(values), means[int(.025 * n)], means[int(.975 * n) - 1]


def grouped(rows: list[dict], lt: float, ltool: float) -> dict[tuple[str, int], list[dict]]:
    out = defaultdict(list)
    for r in rows:
        r = dict(r)
        r["reward"] = reward(r, lt, ltool)
        out[(str(r["task_id"]), int(r["seed"]))].append(r)
    return out


def aggregate(rows: list[dict], lt: float, ltool: float) -> dict[str, Any]:
    by_h = defaultdict(list)
    for r in rows:
        by_h[r["harness"]].append(r)
    result = {}
    for h, rr in sorted(by_h.items()):
        result[h] = {"n": len(rr), "success": statistics.fmean(float(x["success"]) for x in rr),
                     "reward": statistics.fmean(reward(x, lt, ltool) for x in rr),
                     "tokens": statistics.fmean(x["tokens"] for x in rr),
                     "tool_calls": statistics.fmean(x["tool_calls"] for x in rr),
                     "latency_s_median": statistics.median(x["latency_s"] for x in rr),
                     "latency_s_p90": sorted(x["latency_s"] for x in rr)[max(0, math.ceil(.9 * len(rr)) - 1)]}
    return result


def winners(rows: list[dict], lt: float, ltool: float) -> dict[str, Any]:
    tasks = grouped(rows, lt, ltool)
    if not tasks:
        return {}
    harnesses = {r["harness"] for r in rows}
    complete = [rs for rs in tasks.values() if {r["harness"] for r in rs} == harnesses and
                all(sum(r["harness"] == h for r in rs) == 1 for h in harnesses)]
    if not complete:
        raise ValueError("no complete (task_id, seed) harness matrix")
    fixed = {h: statistics.fmean(r["reward"] for rs in complete for r in rs if r["harness"] == h)
             for h in harnesses}
    fixed_h = max(fixed, key=fixed.get)
    gaps, practical, unique = [], Counter(), Counter()
    for rs in complete:
        scores = {r["harness"]: r["reward"] for r in rs}
        if not scores:
            continue
        mx = max(scores.values())
        gaps.append(mx - scores.get(fixed_h, mx))
        best = [h for h, v in scores.items() if abs(v - mx) <= 1e-12]
        for h, v in scores.items():
            if mx - v <= PRACTICAL_REWARD:
                practical[h] += 1
        if len(best) == 1:
            unique[best[0]] += 1
    n = len(complete)
    return {"best_fixed": fixed_h, "best_fixed_reward": fixed[fixed_h],
            "mean_oracle_gap": statistics.fmean(gaps) if gaps else math.nan,
            "oracle_gap_ci95": bootstrap(gaps, 20261008), "n_task_seed_units": n,
            "unique_best_share": {h: unique[h] / n for h in sorted(fixed)},
            "practically_best_share": {h: practical[h] / n for h in sorted(fixed)},
            "practical_threshold_reward": PRACTICAL_REWARD}


def seed_stability(rows: list[dict], lt: float, ltool: float) -> dict[str, Any]:
    per = defaultdict(list)
    for seed in sorted({r["seed"] for r in rows}):
        subset = [r for r in rows if r["seed"] == seed]
        a = aggregate(subset, lt, ltool)
        per[str(seed)] = {h: a[h]["reward"] for h in a}
    orders = [tuple(sorted(x, key=x.get, reverse=True)) for x in per.values() if x]
    consensus = Counter(orders).most_common(1)[0][0] if orders else []
    return {"n_seeds": len(per), "reward_by_seed": per,
            "consensus_order": list(consensus),
            "order_agreement_share": (sum(x == consensus for x in orders) / len(orders)) if orders else math.nan}


def feature_matrix(rows: list[dict]):
    keys = sorted({k for r in rows for k, v in r["task_features"].items() if isinstance(v, (int, float, bool))})
    X = [[float(r["task_features"].get(k, 0)) for k in keys] for r in rows]
    return X, keys


def predictive(rows: list[dict], lt: float, ltool: float, seed: int) -> dict[str, Any]:
    """GroupKFold predictors; returns an explicit unavailable result when sklearn is absent."""
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        return {"status": "unavailable", "reason": "scikit-learn is not installed"}
    # Only non-heldout rows can train/select. One row per task-harness is expected.
    train = [r for r in rows if r["split"] != "heldout"]
    tasks = grouped(train, lt, ltool)
    all_harnesses = {x["harness"] for x in train}
    complete = [rs for rs in tasks.values() if {r["harness"] for r in rs} == all_harnesses and
                all(sum(r["harness"] == h for r in rs) == 1 for h in all_harnesses)]
    if len(complete) < 2:
        return {"status": "unavailable", "reason": "fewer than two complete task groups"}
    hs = sorted({r["harness"] for rs in complete for r in rs})
    # Aggregate one feature vector per task and derive the label from observed outcomes.
    canonical = [next(r for r in rs if r["harness"] == hs[0]) for rs in complete]
    X, keys = feature_matrix(canonical)
    y = [max(hs, key=lambda h: next(r["reward"] for r in rs if r["harness"] == h)) for rs in complete]
    groups = [str(c["task_template"]) for c in canonical]
    n_groups = len(set(groups))
    if n_groups < 2:
        return {"status": "unavailable", "reason": "fewer than two task_template groups"}
    splitter = GroupKFold(n_splits=min(5, n_groups))
    pred, tree_pred, proba, truth, pair_pred, pair_truth, selected_rewards = [], [], [], [], [], [], []
    for tr, te in splitter.split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=seed))
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=seed)
        # Logistic can fail if a fold has one class; use tree/majority fallback for that fold.
        fit = clf if len(set(y[i] for i in tr)) > 1 else tree
        fit.fit([X[i] for i in tr], [y[i] for i in tr])
        pred.extend(fit.predict([X[i] for i in te]))
        tree.fit([X[i] for i in tr], [y[i] for i in tr])
        tree_pred.extend(tree.predict([X[i] for i in te]))
        if hasattr(fit, "predict_proba"):
            pp, classes = fit.predict_proba([X[i] for i in te]), list(fit.classes_)
            for rowp in pp:
                proba.append(max((float(v), c) for v, c in zip(rowp, classes)))
        else:
            proba.extend([(1.0, x) for x in fit.predict([X[i] for i in te])])
        truth.extend(y[i] for i in te)
        selected_rewards.extend(max(next(r["reward"] for r in complete[i] if r["harness"] == h) for h in [p])
                                for i, p in zip(te, fit.predict([X[i] for i in te])))
        # Pairwise expected reward models: fit one ridge per harness, same folds/groups.
        for h in hs:
            yy = [next(r["reward"] for r in complete[i] if r["harness"] == h) for i in tr]
            model = make_pipeline(StandardScaler(), Ridge(alpha=1))
            model.fit([X[i] for i in tr], yy)
            pp = model.predict([X[i] for i in te])
            pair_pred.extend((i, h, float(v)) for i, v in zip(te, pp))
            pair_truth.extend((i, h, next(r["reward"] for r in complete[i] if r["harness"] == h)) for i in te)
    fixed_means = {h: statistics.fmean(next(r["reward"] for r in rs if r["harness"] == h)
                                      for rs in complete) for h in hs}
    best_fixed = max(fixed_means, key=fixed_means.get)
    fixed_reward = [next(r["reward"] for r in rs if r["harness"] == best_fixed) for rs in complete]
    actual = [max(r["reward"] for r in rs) for rs in complete]
    # ECE uses fixed ten bins; probabilities are best-class probabilities.
    ece = 0.0
    for lo in [i / 10 for i in range(10)]:
        ids = [i for i, (p, _) in enumerate(proba) if lo <= p < lo + .1 or (lo == .9 and p == 1)]
        if ids:
            ece += len(ids) / len(proba) * abs(statistics.fmean(proba[i][0] for i in ids) -
                                               statistics.fmean(truth[i] == proba[i][1] for i in ids))
    return {"status": "ok", "features": keys, "n_tasks": len(complete), "n_template_groups": n_groups,
            "logistic_accuracy": accuracy_score(truth, pred), "tree_accuracy": accuracy_score(truth, tree_pred),
            "reward_selected_mean": statistics.fmean(selected_rewards),
            "best_fixed_mean": statistics.fmean(fixed_reward), "oracle_mean": statistics.fmean(actual),
            "ece": ece, "brier_not_applicable_multiclass": True,
            "pair_model_mae": statistics.fmean(abs(a[2] - b[2]) for a, b in zip(pair_pred, pair_truth))}


def render(report: dict) -> str:
    lines = ["# Signal analysis report (synthetic or supplied JSONL)", "", f"Input rows: {report['n_rows']}; tasks: {report['n_tasks']}",
             "", "## Main reward", "", "| harness | success | reward | tokens | tools |", "|---|---:|---:|---:|---:|"]
    for h, x in report["main"]["aggregate"].items():
        lines.append(f"| {h} | {x['success']:.3f} | {x['reward']:.3f} | {x['tokens']:.1f} | {x['tool_calls']:.2f} |")
    w = report["main"]["winners"]
    lines += ["", f"Mean oracle gap: **{w['mean_oracle_gap']:.3f}** (95% bootstrap CI {w['oracle_gap_ci95'][1]:.3f}, {w['oracle_gap_ci95'][2]:.3f}).",
              "", "## Sensitivity and stability", ""]
    for name, x in report["sensitivity"].items():
        lines.append(f"* `{name}`: best fixed `{x['winners']['best_fixed']}`, oracle gap `{x['winners']['mean_oracle_gap']:.3f}`")
    lines += ["", f"Seeds: {report['main']['seed_stability']['n_seeds']}; order agreement: {report['main']['seed_stability']['order_agreement_share']:.3f}.",
              "", "## Predictability", "", "```json", json.dumps(report["predictive"], ensure_ascii=False, indent=2), "```", "", 
              "Held-out rows, if present, were scored after fitting and were not used for model selection.", ""]
    return "\n".join(lines)


def analyze(rows: list[dict], seed: int = 20261008) -> dict:
    # No heldout row is passed to aggregate/winners/predictive until this point.
    analysis_rows = [r for r in rows if r["split"] != "heldout"]
    if not analysis_rows:
        raise ValueError("no non-heldout rows available for fitting/analysis")
    main_agg = aggregate(analysis_rows, .10, .02)
    report = {"n_rows": len(rows), "n_rows_nonheldout": len(analysis_rows),
              "n_tasks": len({r["task_id"] for r in analysis_rows}), "main": {
                  "aggregate": main_agg, "winners": winners(analysis_rows, .10, .02),
                  "seed_stability": seed_stability(analysis_rows, .10, .02)},
              "sensitivity": {}, "predictive": predictive(analysis_rows, .10, .02, seed),
              "config": {"seed": seed, "practical_reward": PRACTICAL_REWARD,
                         "bootstrap_replicates": BOOTSTRAPS, "heldout_used_for_fit": False}}
    for name, lt, ltool in LAMBDAS[1:]:
        report["sensitivity"][name] = {"aggregate": aggregate(analysis_rows, lt, ltool),
                                       "winners": winners(analysis_rows, lt, ltool)}
    # Held-out quality is reported only as a final score, never fed back into fit.
    heldout = [r for r in rows if r["split"] == "heldout"]
    if heldout:
        report["heldout_quality"] = aggregate(heldout, .10, .02)
        report["heldout_n_tasks"] = len({r["task_id"] for r in heldout})
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--output", type=Path, required=True, help="Markdown report path; never overwritten")
    ap.add_argument("--json-output", type=Path)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if (a.output.exists() or (a.json_output and a.json_output.exists())) and not a.overwrite:
        raise SystemExit("refusing to overwrite existing report; pass --overwrite explicitly")
    report = analyze(read_jsonl(a.input), a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(render(report), encoding="utf-8")
    if a.json_output:
        a.json_output.parent.mkdir(parents=True, exist_ok=True)
        a.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(render(report))


def _json_default(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    raise TypeError(type(x).__name__)


if __name__ == "__main__":
    main()
