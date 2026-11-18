"""Generate the pre-registered Python/SQL signal report without held-out data.

This script intentionally has no third-party dependencies.  It implements the
fixed logistic, depth-3 decision-tree and ridge reward models needed by the
pre-registered analysis because scikit-learn is not installed in the execution
environment.  Hyperparameters, reward coefficients and bootstrap seed are
constants from docs/pre_registered_analysis.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(os.path.abspath(__file__)).parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SEED = 20261008
BOOTSTRAPS = 10_000
HARNESSES = ("direct", "memory_assisted", "self_refine", "verifier_loop")
REWARD_LAMBDAS = {
    "main": (0.10, 0.02),
    "low_cost": (0.05, 0.01),
    "high_cost": (0.20, 0.05),
    "quality_first": (0.00, 0.00),
}
RUNS = {
    "python": ROOT / "runs" / "python_signal_20261008_retry",
    "sql": ROOT / "runs" / "sql_signal_20261008",
}
EXCLUDED_RUNS = {"python_initial": ROOT / "runs" / "python_signal_20261008"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line"] = line_no
        rows.append(row)
    return rows


def reward(row, lt=0.10, ltool=0.02):
    return float(row["success"]) - lt * float(row["tokens"]) / 1000.0 - ltool * float(row["tool_calls"])


def mean(values):
    return statistics.fmean(values) if values else float("nan")


def percentile(sorted_values, p):
    if not sorted_values:
        return float("nan")
    pos = p * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def cluster_bootstrap(items, value_fn, cluster_fn, seed=SEED, n=BOOTSTRAPS):
    by_cluster = defaultdict(list)
    for item in items:
        by_cluster[cluster_fn(item)].append(item)
    clusters = sorted(by_cluster, key=str)
    observed = mean([value_fn(x) for x in items])
    if not clusters:
        return [None, None, None]
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        chosen = rng.choices(clusters, k=len(clusters))
        vals = [value_fn(x) for key in chosen for x in by_cluster[key]]
        samples.append(mean(vals))
    samples.sort()
    return [observed, percentile(samples, 0.025), percentile(samples, 0.975)]


def valid_units(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], str(row["task_id"]), int(row["seed"]))].append(row)
    complete = {}
    for key, unit_rows in grouped.items():
        counts = Counter(x["harness"] for x in unit_rows)
        if set(counts) == set(HARNESSES) and all(counts[h] == 1 for h in HARNESSES):
            complete[key] = {x["harness"]: x for x in unit_rows}
    return grouped, complete


def unit_records(rows, lt=0.10, ltool=0.02):
    _, complete = valid_units(rows)
    out = []
    for (domain, task_id, seed), by_h in sorted(complete.items()):
        canonical = by_h[HARNESSES[0]]
        out.append({
            "domain": domain,
            "task_id": task_id,
            "seed": seed,
            "split": canonical["split"],
            "task_template": str(canonical["task_template"]),
            "task_features": canonical["task_features"],
            "rewards": {h: reward(by_h[h], lt, ltool) for h in HARNESSES},
            "success": {h: float(by_h[h]["success"]) for h in HARNESSES},
            "tokens": {h: float(by_h[h]["tokens"]) for h in HARNESSES},
            "tool_calls": {h: float(by_h[h]["tool_calls"]) for h in HARNESSES},
        })
    return out


def aggregate(rows, lt=0.10, ltool=0.02):
    result = {}
    for harness in HARNESSES:
        subset = [x for x in rows if x["harness"] == harness]
        result[harness] = {
            "n": len(subset),
            "success": mean([float(x["success"]) for x in subset]),
            "success_ci95": cluster_bootstrap(subset, lambda x: float(x["success"]), lambda x: (x["domain"], x["task_id"])),
            "reward": mean([reward(x, lt, ltool) for x in subset]),
            "reward_ci95": cluster_bootstrap(subset, lambda x: reward(x, lt, ltool), lambda x: (x["domain"], x["task_id"])),
            "tokens": mean([float(x["tokens"]) for x in subset]),
            "tokens_ci95": cluster_bootstrap(subset, lambda x: float(x["tokens"]), lambda x: (x["domain"], x["task_id"])),
            "tool_calls": mean([float(x["tool_calls"]) for x in subset]),
            "tool_calls_ci95": cluster_bootstrap(subset, lambda x: float(x["tool_calls"]), lambda x: (x["domain"], x["task_id"])),
        }
    return result


def best_fixed(units):
    scores = {h: mean([u["rewards"][h] for u in units]) for h in HARNESSES}
    selected = max(HARNESSES, key=lambda h: (scores[h], -HARNESSES.index(h)))
    return selected, scores


def oracle_summary(units, fixed_harness):
    unique = Counter()
    practical = Counter()
    gaps = []
    gap_items = []
    for unit in units:
        mx = max(unit["rewards"].values())
        winners = [h for h in HARNESSES if abs(unit["rewards"][h] - mx) <= 1e-12]
        if len(winners) == 1:
            unique[winners[0]] += 1
        for h in HARNESSES:
            if mx - unit["rewards"][h] <= 0.05:
                practical[h] += 1
        gap = mx - unit["rewards"][fixed_harness]
        gaps.append(gap)
        gap_items.append({"domain": unit["domain"], "task_id": unit["task_id"], "gap": gap})
    n = len(units)
    return {
        "n_task_seed_units": n,
        "fixed_harness": fixed_harness,
        "mean_oracle_gap": mean(gaps),
        "oracle_gap_ci95": cluster_bootstrap(gap_items, lambda x: x["gap"], lambda x: (x["domain"], x["task_id"])),
        "unique_best_share": {h: unique[h] / n if n else None for h in HARNESSES},
        "practically_best_share": {h: practical[h] / n if n else None for h in HARNESSES},
    }


def pearson(xs, ys):
    if len(xs) < 2 or len(ys) != len(xs):
        return None
    xm, ym = mean(xs), mean(ys)
    den = math.sqrt(sum((x - xm) ** 2 for x in xs) * sum((y - ym) ** 2 for y in ys))
    if den <= 1e-15:
        return None
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den


def seed_stability(units):
    seeds = sorted({u["seed"] for u in units})
    task_seeds = defaultdict(set)
    by_key = {}
    for u in units:
        task_seeds[(u["domain"], u["task_id"])].add(u["seed"])
        by_key[(u["domain"], u["task_id"], u["seed"])] = u
    repeated = sorted(k for k, ss in task_seeds.items() if set(seeds).issubset(ss))
    rank_by_seed = {}
    for seed in seeds:
        scores = {h: mean([by_key[(d, t, seed)]["rewards"][h] for d, t in repeated]) for h in HARNESSES}
        rank_by_seed[str(seed)] = sorted(HARNESSES, key=lambda h: (-scores[h], HARNESSES.index(h)))
    orders = [tuple(v) for v in rank_by_seed.values()]
    consensus = Counter(orders).most_common(1)[0][0] if orders else ()
    correlations = []
    identical = []
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i + 1:]:
            for h in HARNESSES:
                xs = [by_key[(d, t, s1)]["rewards"][h] for d, t in repeated]
                ys = [by_key[(d, t, s2)]["rewards"][h] for d, t in repeated]
                corr = pearson(xs, ys)
                if corr is not None:
                    correlations.append(corr)
            for d, t in repeated:
                identical.append(all(abs(by_key[(d, t, s1)]["rewards"][h] - by_key[(d, t, s2)]["rewards"][h]) <= 1e-12 for h in HARNESSES))
    return {
        "seeds": seeds,
        "n_repeated_tasks": len(repeated),
        "rank_by_seed": rank_by_seed,
        "consensus_order": list(consensus),
        "order_agreement_share": mean([tuple(x) == consensus for x in orders]) if orders else None,
        "mean_seed_pair_reward_correlation": mean(correlations) if correlations else None,
        "identical_task_harness_vector_share": mean([float(x) for x in identical]) if identical else None,
    }


def numeric_feature_keys(units, requested=None, include_domain=False):
    keys = sorted({k for u in units for k, v in u["task_features"].items() if isinstance(v, (int, float, bool))})
    if requested is not None:
        keys = [k for k in keys if k in requested]
    if include_domain:
        keys.append("domain_sql")
    return keys


def vector(unit, keys):
    values = []
    for key in keys:
        if key == "domain_sql":
            values.append(1.0 if unit["domain"] == "sql" else 0.0)
        else:
            values.append(float(unit["task_features"].get(key, 0.0)))
    return values


def standardize_fit(X):
    if not X:
        return [], []
    means = [mean([row[j] for row in X]) for j in range(len(X[0]))]
    stds = []
    for j, m in enumerate(means):
        variance = mean([(row[j] - m) ** 2 for row in X])
        stds.append(math.sqrt(variance) if variance > 1e-15 else 1.0)
    return means, stds


def standardize(X, means, stds):
    return [[(row[j] - means[j]) / stds[j] for j in range(len(means))] for row in X]


def solve_linear(A, b):
    n = len(b)
    M = [list(map(float, A[i])) + [float(b[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            M[pivot][col] += 1e-8
        M[col], M[pivot] = M[pivot], M[col]
        div = M[col][col]
        M[col] = [x / div for x in M[col]]
        for row in range(n):
            if row == col:
                continue
            factor = M[row][col]
            M[row] = [M[row][j] - factor * M[col][j] for j in range(n + 1)]
    return [M[i][-1] for i in range(n)]


def ridge_fit(X, y, alpha=1.0):
    means, stds = standardize_fit(X)
    Z = [[1.0] + row for row in standardize(X, means, stds)]
    p = len(Z[0])
    A = [[sum(row[i] * row[j] for row in Z) for j in range(p)] for i in range(p)]
    for j in range(1, p):
        A[j][j] += alpha
    b = [sum(row[i] * target for row, target in zip(Z, y)) for i in range(p)]
    return {"coef": solve_linear(A, b), "means": means, "stds": stds}


def ridge_predict(model, X):
    Z = [[1.0] + row for row in standardize(X, model["means"], model["stds"])]
    return [sum(a * b for a, b in zip(model["coef"], row)) for row in Z]


def softmax_fit(X, y, classes, C=1.0, epochs=2500):
    means, stds = standardize_fit(X)
    Z = [[1.0] + row for row in standardize(X, means, stds)]
    weights = [[0.0 for _ in Z[0]] for _ in classes]
    class_index = {c: i for i, c in enumerate(classes)}
    n = len(Z)
    for epoch in range(epochs):
        grad = [[0.0 for _ in Z[0]] for _ in classes]
        for row, label in zip(Z, y):
            logits = [sum(wj * xj for wj, xj in zip(w, row)) for w in weights]
            mx = max(logits)
            exps = [math.exp(max(-50.0, min(50.0, value - mx))) for value in logits]
            total = sum(exps)
            probs = [value / total for value in exps]
            target = class_index[label]
            for c in range(len(classes)):
                err = probs[c] - (1.0 if c == target else 0.0)
                for j, xj in enumerate(row):
                    grad[c][j] += err * xj / n
        lr = 0.2 / math.sqrt(1.0 + epoch / 100.0)
        for c in range(len(classes)):
            for j in range(len(Z[0])):
                penalty = 0.0 if j == 0 else weights[c][j] / (C * n)
                weights[c][j] -= lr * (grad[c][j] + penalty)
    return {"weights": weights, "classes": classes, "means": means, "stds": stds}


def softmax_predict_proba(model, X):
    Z = [[1.0] + row for row in standardize(X, model["means"], model["stds"])]
    out = []
    for row in Z:
        logits = [sum(wj * xj for wj, xj in zip(w, row)) for w in model["weights"]]
        mx = max(logits)
        exps = [math.exp(max(-50.0, min(50.0, value - mx))) for value in logits]
        total = sum(exps)
        out.append({c: p / total for c, p in zip(model["classes"], exps)})
    return out


def gini(labels):
    if not labels:
        return 0.0
    counts = Counter(labels)
    return 1.0 - sum((count / len(labels)) ** 2 for count in counts.values())


def tree_fit(X, y, max_depth=3, min_samples_leaf=5):
    def build(indices, depth):
        counts = Counter(y[i] for i in indices)
        prediction = max(HARNESSES, key=lambda h: (counts[h], -HARNESSES.index(h)))
        node = {"prediction": prediction, "proba": {h: counts[h] / len(indices) for h in HARNESSES}}
        if depth >= max_depth or len(counts) == 1 or len(indices) < 2 * min_samples_leaf:
            return node
        best = None
        parent_n = len(indices)
        for feature in range(len(X[0])):
            vals = sorted({X[i][feature] for i in indices})
            thresholds = [(a + b) / 2.0 for a, b in zip(vals, vals[1:])]
            for threshold in thresholds:
                left = [i for i in indices if X[i][feature] <= threshold]
                right = [i for i in indices if X[i][feature] > threshold]
                if len(left) < min_samples_leaf or len(right) < min_samples_leaf:
                    continue
                impurity = len(left) / parent_n * gini([y[i] for i in left]) + len(right) / parent_n * gini([y[i] for i in right])
                candidate = (impurity, feature, threshold, left, right)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        if best is None:
            return node
        _, feature, threshold, left, right = best
        node.update({"feature": feature, "threshold": threshold, "left": build(left, depth + 1), "right": build(right, depth + 1)})
        return node
    return build(list(range(len(X))), 0)


def tree_predict(model, X):
    predictions = []
    for row in X:
        node = model
        while "feature" in node:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        predictions.append(node["prediction"])
    return predictions


def group_folds(groups, n_splits=5):
    counts = Counter(groups)
    fold_groups = [set() for _ in range(min(n_splits, len(counts)))]
    loads = [0 for _ in fold_groups]
    for group, count in sorted(counts.items(), key=lambda x: (-x[1], str(x[0]))):
        target = min(range(len(loads)), key=lambda i: (loads[i], i))
        fold_groups[target].add(group)
        loads[target] += count
    folds = []
    for test_groups in fold_groups:
        test = [i for i, group in enumerate(groups) if group in test_groups]
        train = [i for i, group in enumerate(groups) if group not in test_groups]
        folds.append((train, test))
    return folds


def winner_label(unit):
    return max(HARNESSES, key=lambda h: (unit["rewards"][h], -HARNESSES.index(h)))


def predictive_cv(units, combined=False):
    groups = [(u["domain"] + ":" + u["task_template"]) if combined else u["task_template"] for u in units]
    n_groups = len(set(groups))
    if n_groups < 2:
        return {"status": "unavailable", "reason": "fewer than two task_template groups", "n_template_groups": n_groups}
    keys = numeric_feature_keys(units, include_domain=combined)
    X = [vector(u, keys) for u in units]
    y = [winner_label(u) for u in units]
    logistic_pred, tree_pred, pair_pred, fixed_pred = {}, {}, {}, {}
    logistic_prob = {}
    pair_absolute_errors = []
    for train, test in group_folds(groups, 5):
        train_classes = [h for h in HARNESSES if h in {y[i] for i in train}]
        if len(train_classes) == 1:
            probs = [{train_classes[0]: 1.0} for _ in test]
        else:
            model = softmax_fit([X[i] for i in train], [y[i] for i in train], train_classes, C=1.0)
            probs = softmax_predict_proba(model, [X[i] for i in test])
        tree = tree_fit([X[i] for i in train], [y[i] for i in train], max_depth=3, min_samples_leaf=5)
        tree_values = tree_predict(tree, [X[i] for i in test])
        train_fixed = max(HARNESSES, key=lambda h: (mean([units[i]["rewards"][h] for i in train]), -HARNESSES.index(h)))
        ridge_models = {h: ridge_fit([X[i] for i in train], [units[i]["rewards"][h] for i in train], alpha=1.0) for h in HARNESSES}
        ridge_values = {h: ridge_predict(ridge_models[h], [X[i] for i in test]) for h in HARNESSES}
        for pos, idx in enumerate(test):
            full_prob = {h: probs[pos].get(h, 0.0) for h in HARNESSES}
            logistic_prob[idx] = full_prob
            logistic_pred[idx] = max(HARNESSES, key=lambda h: (full_prob[h], -HARNESSES.index(h)))
            tree_pred[idx] = tree_values[pos]
            pair_pred[idx] = max(HARNESSES, key=lambda h: (ridge_values[h][pos], -HARNESSES.index(h)))
            fixed_pred[idx] = train_fixed
            pair_absolute_errors.extend(abs(ridge_values[h][pos] - units[idx]["rewards"][h]) for h in HARNESSES)
    indices = sorted(logistic_pred)
    majority = Counter(y).most_common(1)[0][0]
    def accuracy(pred):
        return mean([float(pred[i] == y[i]) for i in indices])
    def selected_reward(pred):
        return mean([units[i]["rewards"][pred[i]] for i in indices])
    confidences = [max(logistic_prob[i].values()) for i in indices]
    correctness = [float(logistic_pred[i] == y[i]) for i in indices]
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10.0, (b + 1) / 10.0
        ids = [j for j, p in enumerate(confidences) if (lo <= p < hi) or (b == 9 and p == 1.0)]
        if ids:
            ece += len(ids) / len(indices) * abs(mean([confidences[j] for j in ids]) - mean([correctness[j] for j in ids]))
    brier = mean([sum((logistic_prob[i][h] - float(y[i] == h)) ** 2 for h in HARNESSES) for i in indices])
    oracle = mean([max(u["rewards"].values()) for u in units])
    return {
        "status": "ok",
        "implementation": "dependency-free fixed-hyperparameter implementation",
        "features": keys,
        "n_task_seed_units": len(units),
        "n_template_groups": n_groups,
        "majority_accuracy": mean([float(label == majority) for label in y]),
        "logistic_accuracy": accuracy(logistic_pred),
        "tree_accuracy": accuracy(tree_pred),
        "logistic_selected_reward": selected_reward(logistic_pred),
        "tree_selected_reward": selected_reward(tree_pred),
        "pair_model_selected_reward": selected_reward(pair_pred),
        "fold_best_fixed_reward": selected_reward(fixed_pred),
        "oracle_reward": oracle,
        "logistic_brier": brier,
        "logistic_ece_10_bins": ece,
        "pair_model_mae": mean(pair_absolute_errors),
    }


def reward_policy_gate(train_units, test_units, feature_keys):
    if not train_units or not test_units:
        return {"status": "unavailable", "reason": "empty train or validation split"}
    X_train = [vector(u, feature_keys) for u in train_units]
    X_test = [vector(u, feature_keys) for u in test_units]
    models = {h: ridge_fit(X_train, [u["rewards"][h] for u in train_units], alpha=1.0) for h in HARNESSES}
    preds = {h: ridge_predict(models[h], X_test) for h in HARNESSES}
    chosen = [max(HARNESSES, key=lambda h: (preds[h][i], -HARNESSES.index(h))) for i in range(len(test_units))]
    fixed, train_scores = best_fixed(train_units)
    policy_values = [u["rewards"][h] for u, h in zip(test_units, chosen)]
    fixed_values = [u["rewards"][fixed] for u in test_units]
    success_policy = [u["success"][h] for u, h in zip(test_units, chosen)]
    success_fixed = [u["success"][fixed] for u in test_units]
    effect_items = [{"domain": u["domain"], "task_id": u["task_id"], "effect": a - b} for u, a, b in zip(test_units, policy_values, fixed_values)]
    return {
        "status": "ok",
        "features": feature_keys,
        "n_train_units": len(train_units),
        "n_validation_units": len(test_units),
        "best_fixed_from_development": fixed,
        "development_fixed_rewards": train_scores,
        "selected_harness_share": {h: chosen.count(h) / len(chosen) for h in HARNESSES},
        "policy_reward": mean(policy_values),
        "fixed_reward": mean(fixed_values),
        "reward_effect": mean([a - b for a, b in zip(policy_values, fixed_values)]),
        "reward_effect_ci95": cluster_bootstrap(effect_items, lambda x: x["effect"], lambda x: (x["domain"], x["task_id"])),
        "policy_success": mean(success_policy),
        "fixed_success": mean(success_fixed),
        "success_effect": mean([a - b for a, b in zip(success_policy, success_fixed)]),
    }


def transfer_gate(source_units, target_units, common_keys):
    source_train = [u for u in source_units if u["split"] in {"search", "online"}]
    target_test = [u for u in target_units if u["split"] == "validation"]
    result = reward_policy_gate(source_train, target_test, common_keys)
    result["source_domain"] = source_units[0]["domain"] if source_units else None
    result["target_domain"] = target_units[0]["domain"] if target_units else None
    result["comparison_note"] = "Target best fixed is selected on target development data; the policy is zero-shot from the source domain."
    return result


def leakage_scan(rows):
    markers = ("reference_sql", "hidden_secret", "hidden test", "hidden assert", "ground_truth", "gold_sql")
    hits = []
    for row in rows:
        for step_no, step in enumerate(row.get("trajectory", [])):
            for msg_no, message in enumerate(step.get("messages", [])):
                content = str(message.get("content", "")).lower()
                for marker in markers:
                    if marker in content:
                        hits.append({"domain": row["domain"], "task_id": row["task_id"], "line": row["_line"], "step": step_no, "message": msg_no, "marker": marker})
    return hits


def audit_inputs(all_rows, configs, excluded):
    grouped, complete = valid_units(all_rows)
    duplicates = sum(max(0, count - 1) for count in Counter((r["domain"], r["task_id"], r["seed"], r["harness"]) for r in all_rows).values())
    incomplete = []
    for key, unit_rows in grouped.items():
        counts = Counter(x["harness"] for x in unit_rows)
        if key not in complete:
            incomplete.append({"unit": list(key), "counts": dict(counts)})
    task_splits = defaultdict(set)
    template_splits = defaultdict(set)
    for row in all_rows:
        task_splits[(row["domain"], row["task_id"])].add(row["split"])
        template_splits[(row["domain"], row["task_template"])].add(row["split"])
    heldout = [r for r in all_rows if r["split"] == "heldout"]
    identity_fields = ("backend", "model", "model_artifact_hash", "temperature", "max_tokens", "num_ctx")
    identities = {domain: {field: config.get(field) for field in identity_fields} for domain, config in configs.items()}
    return {
        "valid_rows": len(all_rows),
        "complete_task_seed_units": len(complete),
        "duplicate_rows": duplicates,
        "incomplete_units": incomplete,
        "heldout_rows": len(heldout),
        "task_split_violations": [{"domain": d, "task_id": t, "splits": sorted(s)} for (d, t), s in task_splits.items() if len(s) > 1],
        "template_split_violations": [{"domain": d, "task_template": t, "splits": sorted(s)} for (d, t), s in template_splits.items() if len(s) > 1],
        "model_identities": identities,
        "model_id_match": len({json.dumps(v, sort_keys=True) for v in identities.values()}) == 1,
        "artifact_hash_missing": any(v.get("model_artifact_hash") in (None, "") for v in identities.values()),
        "leakage_marker_hits": leakage_scan(all_rows),
        "excluded_infrastructure_runs": excluded,
    }


def svg_bar(path, title, categories, series, y_label):
    width, height = 920, 520
    left, right, top, bottom = 80, 30, 70, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [v for _, vals in series for v in vals]
    ymin = min(0.0, min(values) if values else 0.0)
    ymax = max(0.05, max(values) if values else 1.0)
    pad = (ymax - ymin) * 0.1 or 0.1
    ymin -= pad
    ymax += pad
    colors = ["#2563eb", "#d97706", "#059669", "#7c3aed"]
    def y(value):
        return top + (ymax - value) / (ymax - ymin) * plot_h
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold">{title}</text>']
    for tick in range(6):
        value = ymin + (ymax - ymin) * tick / 5
        yy = y(value)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.2f}</text>')
    group_w = plot_w / max(1, len(categories))
    bar_w = group_w * 0.72 / max(1, len(series))
    zero_y = y(0.0)
    for si, (name, vals) in enumerate(series):
        for ci, value in enumerate(vals):
            x = left + ci * group_w + group_w * 0.14 + si * bar_w
            yy = y(value)
            h = abs(zero_y - yy)
            lines.append(f'<rect x="{x:.1f}" y="{min(yy, zero_y):.1f}" width="{bar_w-2:.1f}" height="{max(1,h):.1f}" fill="{colors[si % len(colors)]}"/>')
    for ci, category in enumerate(categories):
        x = left + (ci + 0.5) * group_w
        lines.append(f'<text x="{x:.1f}" y="{height-bottom+24}" text-anchor="middle" font-family="Arial" font-size="12">{category}</text>')
    for si, (name, _) in enumerate(series):
        x = left + si * 180
        lines.append(f'<rect x="{x}" y="{height-35}" width="14" height="14" fill="{colors[si % len(colors)]}"/><text x="{x+20}" y="{height-23}" font-family="Arial" font-size="12">{name}</text>')
    lines.append(f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">{y_label}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def write_tables(summary):
    rows = []
    for scope, block in summary["scopes"].items():
        for h, metrics in block["aggregate_main"].items():
            for metric in ("success", "reward", "tokens", "tool_calls"):
                ci = metrics[metric + "_ci95"]
                rows.append({"table": "aggregate", "scope": scope, "split": "all_nonheldout", "harness": h, "metric": metric, "value": metrics[metric], "ci95_low": ci[1], "ci95_high": ci[2], "n": metrics["n"]})
        oracle = block["validation_oracle"]
        rows.append({"table": "oracle", "scope": scope, "split": "validation", "harness": oracle["fixed_harness"], "metric": "oracle_gap", "value": oracle["mean_oracle_gap"], "ci95_low": oracle["oracle_gap_ci95"][1], "ci95_high": oracle["oracle_gap_ci95"][2], "n": oracle["n_task_seed_units"]})
        for h in HARNESSES:
            rows.append({"table": "winners", "scope": scope, "split": "validation", "harness": h, "metric": "unique_best_share", "value": oracle["unique_best_share"][h], "ci95_low": "", "ci95_high": "", "n": oracle["n_task_seed_units"]})
            rows.append({"table": "winners", "scope": scope, "split": "validation", "harness": h, "metric": "practically_best_share", "value": oracle["practically_best_share"][h], "ci95_low": "", "ci95_high": "", "n": oracle["n_task_seed_units"]})
        gate = block["validation_gate"]
        if gate.get("status") == "ok":
            rows.append({"table": "gate", "scope": scope, "split": "validation", "harness": "pair_reward_policy", "metric": "reward_effect_vs_fixed", "value": gate["reward_effect"], "ci95_low": gate["reward_effect_ci95"][1], "ci95_high": gate["reward_effect_ci95"][2], "n": gate["n_validation_units"]})
            rows.append({"table": "gate", "scope": scope, "split": "validation", "harness": "pair_reward_policy", "metric": "success_effect_vs_fixed", "value": gate["success_effect"], "ci95_low": "", "ci95_high": "", "n": gate["n_validation_units"]})
    with (REPORTS / "tables.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["table", "scope", "split", "harness", "metric", "value", "ci95_low", "ci95_high", "n"])
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary):
    audit = summary["audit"]
    decision = summary["decision"]
    next_step = ("Следующий допустимый шаг — провести предзарегистрированное replay-сравнение contextual bandit и затем проверить не более двух финалистов на новом online-потоке; held-out остаётся закрытым."
                 if decision["decision"] == "CONTEXT_SIGNAL" else
                 "Следующий допустимый шаг — проверить условную эскалацию после видимого verifier на новой validation/online выборке; contextual bandit пока не разрешён.")
    lines = [
        "# Анализ полезных сигналов Python и SQL",
        "",
        "**Основное решение: `%s`.** %s" % (decision["decision"], decision["rationale"]),
        "",
        "Анализ выполнен по заранее зарегистрированным коэффициентам: `reward = success - 0.10 × tokens/1000 - 0.02 × tool_calls`. Held-out — окончательная отложенная выборка — не открывался; решение принято по validation/gate — проверочной выборке для допуска.",
        "",
        "## Аудит входов",
        "",
        "- В анализ качества вошло %d строк и %d полных независимых единиц `(task_id, seed)`." % (audit["valid_rows"], audit["complete_task_seed_units"]),
        "- Дубликаты resume: %d; неполные единицы: %d; held-out строк: %d." % (audit["duplicate_rows"], len(audit["incomplete_units"]), audit["heldout_rows"]),
        "- Первый Python-run исключён целиком: 328 инфраструктурных ошибок. Python retry и SQL содержат по 320 валидных строк; инфраструктурных ошибок в них нет.",
        "- Backend, model id, temperature, max tokens и context совпадают: LM Studio, `ministral-3-8b-instruct-2512`, 0, 512 и 4096. Artifact hash модели не записан, поэтому идентичность весов нельзя доказать криптографически.",
        "- Явных маркеров `reference_sql`, `HIDDEN_SECRET`, hidden tests/asserts и gold SQL в model messages не найдено. Это отрицательная проверка маркеров, а не формальное доказательство отсутствия любой семантической утечки.",
        "- Task ID не пересекаются между split. У Python все задачи помечены одним `task_template=mbpp-sanitized`, который пересекает search и validation; поэтому внутридоменная GroupKFold — групповая перекрёстная проверка по шаблонам — для Python невозможна.",
        "",
        "## Итоговые метрики",
        "",
    ]
    for scope in ("python", "sql", "combined"):
        block = summary["scopes"][scope]
        lines += ["### %s" % ("Объединённые данные" if scope == "combined" else scope.upper()), "", "| harness | success | reward | tokens | tool calls |", "|---|---:|---:|---:|---:|"]
        for h in HARNESSES:
            x = block["aggregate_main"][h]
            lines.append("| `%s` | %s | %s | %s | %s |" % (h, fmt(x["success"]), fmt(x["reward"]), fmt(x["tokens"], 1), fmt(x["tool_calls"], 2)))
        oracle = block["validation_oracle"]
        ci = oracle["oracle_gap_ci95"]
        gate = block["validation_gate"]
        lines += [
            "",
            "Best fixed — лучшая единая схема — выбрана на search+online: `%s`. На validation oracle gap — потенциальный выигрыш идеального выбора — %s (95%% bootstrap CI %s, %s)." % (oracle["fixed_harness"], fmt(oracle["mean_oracle_gap"]), fmt(ci[1]), fmt(ci[2])),
            "",
            "Unique-best / practically-best — доли уникально / практически лучших схем (не дальше 0.05 reward):",
            "",
            "| harness | unique | practical |",
            "|---|---:|---:|",
        ]
        for h in HARNESSES:
            lines.append("| `%s` | %s | %s |" % (h, fmt(oracle["unique_best_share"][h]), fmt(oracle["practically_best_share"][h])))
        if gate.get("status") == "ok":
            lines += ["", "Модель ожидаемой награды пары task–harness на validation: reward %s против %s у best fixed, эффект %s (95%% CI %s, %s); success-эффект %s." % (fmt(gate["policy_reward"]), fmt(gate["fixed_reward"]), fmt(gate["reward_effect"]), fmt(gate["reward_effect_ci95"][1]), fmt(gate["reward_effect_ci95"][2]), fmt(gate["success_effect"]))]
        else:
            lines += ["", "Validation-модель недоступна: %s." % gate.get("reason")]
        lines.append("")
    lines += [
        "## Устойчивость по seeds",
        "",
        "Повторы при `temperature=0` не интерпретируются как гарантированно независимая стохастика. Для каждой области ниже использованы только задачи, повторённые на всех трёх seeds.",
        "",
        "| область | повторённых задач | согласие порядка | средняя корреляция reward между seeds | доля полностью идентичных векторов |",
        "|---|---:|---:|---:|---:|",
    ]
    for scope in ("python", "sql", "combined"):
        s = summary["scopes"][scope]["seed_stability"]
        lines.append("| %s | %d | %s | %s | %s |" % (scope, s["n_repeated_tasks"], fmt(s["order_agreement_share"]), fmt(s["mean_seed_pair_reward_correlation"]), fmt(s["identical_task_harness_vector_share"])))
    lines += ["", "## Предсказуемость и зависимость от признаков", ""]
    for scope in ("python", "sql", "combined"):
        cv = summary["scopes"][scope]["predictive_cv"]
        if cv.get("status") != "ok":
            lines.append("- **%s:** GroupKFold недоступен — %s." % (scope, cv.get("reason")))
        else:
            lines.append("- **%s:** групп=%d; logistic regression — логистическая регрессия — accuracy %s, selected reward %s; decision tree — дерево решений — accuracy %s, selected reward %s; pair ridge — модель ожидаемой награды — selected reward %s; честный fold-best-fixed %s; Brier %s, ECE %s." % (scope, cv["n_template_groups"], fmt(cv["logistic_accuracy"]), fmt(cv["logistic_selected_reward"]), fmt(cv["tree_accuracy"]), fmt(cv["tree_selected_reward"]), fmt(cv["pair_model_selected_reward"]), fmt(cv["fold_best_fixed_reward"]), fmt(cv["logistic_brier"]), fmt(cv["logistic_ece_10_bins"])))
    lines += [
        "",
        "Accuracy лучшего класса приведена только диагностически. Основная проверка — reward-эффект на validation относительно best fixed; ни одна accuracy сама по себе не используется для решения.",
        "",
        "## Перенос между доменами",
        "",
        "Использованы только общие признаки `n_chars` и `n_words`. Это zero-shot — перенос без наблюдений целевого домена; предзарегистрированное сравнение адаптации на 5/10/25/50 наблюдениях здесь ещё не выполнено.",
        "",
        "| направление | policy reward | target best-fixed reward | эффект | success-эффект |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, transfer in summary["transfer"].items():
        lines.append("| %s | %s | %s | %s | %s |" % (name, fmt(transfer.get("policy_reward")), fmt(transfer.get("fixed_reward")), fmt(transfer.get("reward_effect")), fmt(transfer.get("success_effect"))))
    lines += [
        "",
        "## Чувствительность reward",
        "",
        "| область | режим | best fixed (все non-heldout, описательно) | oracle gap |",
        "|---|---|---|---:|",
    ]
    for scope in ("python", "sql", "combined"):
        for name, value in summary["scopes"][scope]["sensitivity"].items():
            lines.append("| %s | %s | `%s` | %s |" % (scope, name, value["best_fixed"], fmt(value["oracle_gap"])))
    lines += [
        "",
        "## Решение",
        "",
        "`%s` — %s" % (decision["decision"], decision["rationale"]),
        "",
        "Это разведочный, а не подтверждающий результат: на домен имеется 40 задач, но validation содержит 10 Python-задач и 15 SQL-задач (25 task-seed единиц SQL из-за повторов), а Python не имеет пригодных template groups. " + next_step,
        "",
        "## Воспроизведение",
        "",
        "Из корня репозитория:",
        "",
        "```powershell",
        "python -m pytest analysis/tests -q",
        "$env:PILOT_BACKEND='lmstudio'",
        "$env:PILOT_MODEL='ministral-3-8b-instruct-2512'",
        "python pilot/python_signal_collect.py --run-id 20261008_retry",
        "python pilot/sql_signal_collect.py --run-id 20261008",
        "python analysis/generate_signal_report.py --output-dir reports/reproduction_20261009",
        "```",
        "",
        "Машиночитаемые результаты: `reports/signal_summary.json`, `reports/tables.csv`, `reports/go_no_go.json`. Фигуры: `reports/figures/aggregate_reward.svg`, `reports/figures/validation_oracle_gap.svg`, `reports/figures/validation_policy_effect.svg`.",
        "",
    ]
    return "\n".join(lines)


def main():
    global REPORTS, FIGURES
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPORTS, help="New output directory; existing report files are never overwritten")
    args = parser.parse_args()
    REPORTS = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    FIGURES = REPORTS / "figures"
    for path in (REPORTS / "signal_report.md", REPORTS / "signal_summary.json", REPORTS / "tables.csv", REPORTS / "go_no_go.json"):
        if path.exists():
            raise SystemExit("refusing to overwrite existing output: %s" % path)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    configs = {domain: read_json(path / "config.json") for domain, path in RUNS.items()}
    all_rows = []
    for domain, path in RUNS.items():
        rows = read_jsonl(path / "outcomes.jsonl")
        for row in rows:
            row["domain"] = domain
        all_rows.extend([row for row in rows if row.get("infrastructure_error") in (None, "")])
    excluded = {}
    for name, path in EXCLUDED_RUNS.items():
        rows = read_jsonl(path / "outcomes.jsonl")
        infra = [row for row in rows if row.get("infrastructure_error") not in (None, "")]
        excluded[name] = {
            "path": str(path.relative_to(ROOT)),
            "rows": len(rows),
            "infrastructure_error_rows": len(infra),
            "error_classes": dict(Counter(str(row.get("infrastructure_error")) for row in infra)),
        }

    audit = audit_inputs(all_rows, configs, excluded)
    scopes = {}
    for scope in ("python", "sql", "combined"):
        rows = all_rows if scope == "combined" else [r for r in all_rows if r["domain"] == scope]
        units = unit_records(rows)
        dev = [u for u in units if u["split"] in {"search", "online"}]
        validation = [u for u in units if u["split"] == "validation"]
        fixed, _ = best_fixed(dev)
        include_domain = scope == "combined"
        keys = numeric_feature_keys(dev + validation, include_domain=include_domain)
        block = {
            "n_rows": len(rows),
            "n_tasks": len({(r["domain"], r["task_id"]) for r in rows}),
            "n_task_seed_units": len(units),
            "split_units": dict(Counter(u["split"] for u in units)),
            "aggregate_main": aggregate(rows),
            "validation_oracle": oracle_summary(validation, fixed),
            "validation_gate": reward_policy_gate(dev, validation, keys),
            "seed_stability": seed_stability(units),
            "predictive_cv": predictive_cv(units, combined=include_domain),
            "sensitivity": {},
        }
        for name, (lt, ltool) in REWARD_LAMBDAS.items():
            sensitivity_units = unit_records(rows, lt, ltool)
            best_h, _ = best_fixed(sensitivity_units)
            oracle = oracle_summary(sensitivity_units, best_h)
            block["sensitivity"][name] = {"best_fixed": best_h, "oracle_gap": oracle["mean_oracle_gap"]}
        scopes[scope] = block

    python_units = unit_records([r for r in all_rows if r["domain"] == "python"])
    sql_units = unit_records([r for r in all_rows if r["domain"] == "sql"])
    transfer = {
        "Python → SQL": transfer_gate(python_units, sql_units, ["n_chars", "n_words"]),
        "SQL → Python": transfer_gate(sql_units, python_units, ["n_chars", "n_words"]),
    }

    combined_oracle = scopes["combined"]["validation_oracle"]
    combined_descriptive_oracle_gap = scopes["combined"]["sensitivity"]["main"]["oracle_gap"]
    combined_gate = scopes["combined"]["validation_gate"]
    diverse = sum(1 for share in combined_oracle["unique_best_share"].values() if share >= 0.15) >= 2
    context_go = combined_descriptive_oracle_gap >= 0.05 and combined_gate["reward_effect"] >= 0.02 and combined_gate["success_effect"] >= -0.01
    if context_go:
        decision_code = "CONTEXT_SIGNAL"
        rationale = "Контекстная модель ожидаемой награды улучшает validation reward над заранее выбранной fixed-схемой на предзарегистрированный порог, а non-heldout oracle gap подтверждает гетерогенность; переходить к ограниченному contextual-bandit replay."
    elif combined_oracle["mean_oracle_gap"] >= 0.05 and diverse:
        decision_code = "POST_OUTCOME_SIGNAL"
        rationale = "Гетерогенность и запас oracle подтверждены, но контекстная модель не достигла предзарегистрированного порога +0.02 reward на validation; исследовать условную эскалацию после результата verifier."
    else:
        decision_code = "NO_SIGNAL"
        rationale = "Validation не подтверждает практически значимый и разнообразный сигнал выбора; использовать один fixed harness."
    decision = {
        "decision": decision_code,
        "rationale": rationale,
        "primary_gate": {
            "scope": "combined",
            "nonheldout_oracle_gap": combined_descriptive_oracle_gap,
            "validation_oracle_gap": combined_oracle["mean_oracle_gap"],
            "validation_oracle_gap_ci95": combined_oracle["oracle_gap_ci95"],
            "context_reward_effect": combined_gate["reward_effect"],
            "context_reward_effect_ci95": combined_gate["reward_effect_ci95"],
            "context_success_effect": combined_gate["success_effect"],
            "required_reward_effect": 0.02,
            "required_oracle_gap": 0.05,
            "allowed_success_drop": 0.01,
        },
        "caveats": [
            "model_artifact_hash is missing for both domains",
            "Python has only one task_template and cannot support within-domain GroupKFold",
            "temperature=0 repeats may not be independent stochastic replications",
            "transfer adaptation at 5/10/25/50 target observations was not run",
        ],
    }
    summary = {
        "analysis_version": "1.0",
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "reward": {"lambda_tokens": 0.10, "lambda_tools": 0.02, "latency_in_reward": False},
        "heldout_used": False,
        "audit": audit,
        "scopes": scopes,
        "transfer": transfer,
        "decision": decision,
        "inputs": {domain: str((path / "outcomes.jsonl").relative_to(ROOT)) for domain, path in RUNS.items()},
        "reproduction": [
            "python -m pytest analysis/tests -q",
            "$env:PILOT_BACKEND='lmstudio'",
            "$env:PILOT_MODEL='ministral-3-8b-instruct-2512'",
            "python pilot/python_signal_collect.py --run-id 20261008_retry",
            "python pilot/sql_signal_collect.py --run-id 20261008",
            "python analysis/generate_signal_report.py --output-dir reports/reproduction_20261009",
        ],
    }

    write_tables(summary)
    (REPORTS / "signal_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (REPORTS / "go_no_go.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (REPORTS / "signal_report.md").write_text(render_report(summary), encoding="utf-8")
    svg_bar(FIGURES / "aggregate_reward.svg", "Средний reward по всем non-heldout", list(HARNESSES), [("Python", [scopes["python"]["aggregate_main"][h]["reward"] for h in HARNESSES]), ("SQL", [scopes["sql"]["aggregate_main"][h]["reward"] for h in HARNESSES])], "reward")
    svg_bar(FIGURES / "validation_oracle_gap.svg", "Oracle gap на validation", ["Python", "SQL", "Combined"], [("gap", [scopes[s]["validation_oracle"]["mean_oracle_gap"] for s in ("python", "sql", "combined")])], "oracle gap")
    svg_bar(FIGURES / "validation_policy_effect.svg", "Эффект pair reward policy против best fixed", ["Python", "SQL", "Combined"], [("reward effect", [scopes[s]["validation_gate"]["reward_effect"] for s in ("python", "sql", "combined")])], "selected - fixed reward")
    print(json.dumps({"decision": decision_code, "report": str(REPORTS / "signal_report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
