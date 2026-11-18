"""Low-data harness-selection replay on the frozen Python/SQL signal matrices.

This is Agent H's stage-1 experiment.  It intentionally never reads held-out
rows.  Validation rows are used for scoring only and are never supplied to a
policy update.  A replay observation reveals only the outcome of the selected
harness, except for the fixed direct->verifier fallback whose two attempted
harnesses are both charged to its evaluation cost.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(os.path.abspath(__file__)).parents[1]
HARNESSES = ("direct", "memory_assisted", "self_refine", "verifier_loop")
BUDGETS = (0, 5, 10, 25, 50)
BASE_SEED = 20261008
SOURCE_PRETRAIN_BUDGET = 50
RUNS = {
    "python": ROOT / "runs" / "python_signal_20261008_retry" / "outcomes.jsonl",
    "sql": ROOT / "runs" / "sql_signal_20261008" / "outcomes.jsonl",
}


def reward(row):
    return float(row["success"]) - .10 * float(row["tokens"]) / 1000.0 - .02 * float(row["tool_calls"])


def mean(xs):
    return statistics.fmean(xs) if xs else None


def ci95(xs):
    if not xs:
        return [None, None, None]
    m = statistics.fmean(xs)
    if len(xs) == 1:
        return [m, m, m]
    d = 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))
    return [m, m - d, m + d]


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_units():
    rows = []
    input_hashes = {}
    configs = {}
    for domain, path in RUNS.items():
        input_hashes[str(path.relative_to(ROOT))] = file_hash(path)
        cfg_path = path.parent / "config.json"
        configs[domain] = json.loads(cfg_path.read_text(encoding="utf-8"))
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["domain"] = domain
                rows.append(r)
    identities = {(c["backend"], c["model"], c["temperature"], c["max_tokens"], c["num_ctx"])
                  for c in configs.values()}
    if len(identities) != 1:
        raise ValueError(f"mixed target configurations: {sorted(identities)}")
    if any(r.get("split") == "heldout" for r in rows):
        raise ValueError("held-out rows are forbidden in policy-selection replay")
    if any(r.get("infrastructure_error") for r in rows):
        raise ValueError("infrastructure-error rows are forbidden")
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["domain"], str(r["task_id"]), int(r["seed"]))].append(r)
    units = []
    rejected = []
    for key, rr in sorted(grouped.items()):
        by_h = {r["harness"]: r for r in rr}
        if len(rr) != len(HARNESSES) or set(by_h) != set(HARNESSES):
            rejected.append(key)
            continue
        canonical = by_h[HARNESSES[0]]
        units.append({
            "key": list(key), "domain": key[0], "task_id": key[1], "seed": key[2],
            "split": canonical["split"], "task_template": str(canonical["task_template"]),
            "features": canonical["task_features"],
            "outcomes": {h: {"reward": reward(by_h[h]), "success": float(by_h[h]["success"]),
                              "tokens": float(by_h[h]["tokens"]),
                              "tool_calls": float(by_h[h]["tool_calls"])} for h in HARNESSES},
        })
    return units, {"input_hashes": input_hashes, "configs": configs, "rejected_units": rejected,
                   "n_rows": len(rows), "n_complete_units": len(units)}


class Encoder:
    def __init__(self, units, cross_domain=True):
        self.keys = sorted({k for u in units for k, v in u["features"].items()
                            if isinstance(v, (int, float, bool))})
        if cross_domain:
            self.keys.append("domain_sql")
        raw = np.asarray([self.raw(u) for u in units], dtype=float)
        self.mu = raw.mean(axis=0) if len(raw) else np.zeros(len(self.keys))
        sd = raw.std(axis=0) if len(raw) else np.ones(len(self.keys))
        self.sd = np.where(sd > 1e-9, sd, 1.0)

    def raw(self, unit):
        return [float(unit["domain"] == "sql") if k == "domain_sql"
                else float(unit["features"].get(k, 0.0)) for k in self.keys]

    def __call__(self, unit):
        z = (np.asarray(self.raw(unit), dtype=float) - self.mu) / self.sd
        return np.r_[1.0, z]

    def state(self):
        return {"feature_names": ["intercept"] + self.keys,
                "mean": self.mu.round(8).tolist(), "scale": self.sd.round(8).tolist()}


class Policy:
    name = "base"

    def __init__(self, rng, dim):
        self.rng, self.dim = rng, dim
        self.n = np.zeros(len(HARNESSES), dtype=int)
        self.s = np.zeros(len(HARNESSES), dtype=float)
        self.frozen = False

    def argmax(self, values):
        values = np.asarray(values, dtype=float)
        ids = np.flatnonzero(np.isclose(values, values.max()))
        return int(self.rng.choice(ids))

    def select(self, x, t, deploy=False):
        raise NotImplementedError

    def update(self, arm, r, x):
        if self.frozen:
            raise AssertionError("frozen policy cannot be updated from validation labels")
        self.n[arm] += 1
        self.s[arm] += r

    def means(self):
        return np.divide(self.s, np.maximum(self.n, 1))

    def state(self):
        return {"n": self.n.tolist(), "mean_reward": self.means().round(8).tolist()}


class RandomPolicy(Policy):
    name = "random"

    def select(self, x, t, deploy=False):
        return int(self.rng.integers(len(HARNESSES)))


class BestFixed(Policy):
    name = "best_fixed_round_robin"

    def select(self, x, t, deploy=False):
        if not deploy:
            return t % len(HARNESSES)
        if not self.n.any():
            return int(self.rng.integers(len(HARNESSES)))
        return self.argmax(np.where(self.n > 0, self.means(), -np.inf))


class EpsilonGreedy(Policy):
    name = "epsilon_greedy"

    def select(self, x, t, deploy=False):
        unseen = np.flatnonzero(self.n == 0)
        if not deploy and len(unseen):
            return int(unseen[t % len(unseen)])
        if not deploy and self.rng.random() < .1:
            return int(self.rng.integers(len(HARNESSES)))
        if not self.n.any():
            return int(self.rng.integers(len(HARNESSES)))
        return self.argmax(np.where(self.n > 0, self.means(), -np.inf))


class UCB1(Policy):
    name = "ucb1"

    def select(self, x, t, deploy=False):
        unseen = np.flatnonzero(self.n == 0)
        if len(unseen):
            return int(unseen[t % len(unseen)]) if not deploy else int(self.rng.choice(unseen))
        if deploy:
            return self.argmax(self.means())
        bonus = np.sqrt(2.0 * math.log(max(2, int(self.n.sum()))) / self.n)
        return self.argmax(self.means() + bonus)


class Thompson(Policy):
    name = "thompson_sampling"

    def __init__(self, *args):
        super().__init__(*args)
        self.alpha = np.ones(len(HARNESSES))
        self.beta = np.ones(len(HARNESSES))

    def select(self, x, t, deploy=False):
        score = self.alpha / (self.alpha + self.beta) if deploy else self.rng.beta(self.alpha, self.beta)
        return self.argmax(score)

    def update(self, arm, r, x):
        super().update(arm, r, x)
        # Reward bounds are fixed by the registered formula: min >= -0.3 in these data, max <= 1.
        p = float(np.clip((r + .3) / 1.3, 0.0, 1.0))
        hit = self.rng.random() < p
        self.alpha[arm] += float(hit)
        self.beta[arm] += float(not hit)

    def state(self):
        return {**super().state(), "alpha": self.alpha.tolist(), "beta": self.beta.tolist()}


class LinearBase(Policy):
    def __init__(self, *args):
        super().__init__(*args)
        self.A = [np.eye(self.dim) for _ in HARNESSES]
        self.b = [np.zeros(self.dim) for _ in HARNESSES]

    def theta(self, arm):
        return np.linalg.solve(self.A[arm], self.b[arm])

    def update(self, arm, r, x):
        super().update(arm, r, x)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += r * x

    def state(self):
        return {**super().state(), "theta": [self.theta(a).round(8).tolist() for a in range(len(HARNESSES))]}


class LinUCB(LinearBase):
    name = "linucb"

    def select(self, x, t, deploy=False):
        scores = []
        for a in range(len(HARNESSES)):
            inv = np.linalg.inv(self.A[a])
            bonus = 0.0 if deploy else math.sqrt(max(0.0, float(x @ inv @ x)))
            scores.append(float(self.theta(a) @ x) + bonus)
        return self.argmax(scores)


class LinearThompson(LinearBase):
    name = "linear_thompson_sampling"

    def select(self, x, t, deploy=False):
        scores = []
        for a in range(len(HARNESSES)):
            mu = self.theta(a)
            theta = mu if deploy else self.rng.multivariate_normal(mu, .25 * np.linalg.inv(self.A[a]))
            scores.append(float(theta @ x))
        return self.argmax(scores)


class SupervisedRewardModel(LinearBase):
    name = "supervised_reward_model_diagnostic"

    def select(self, x, t, deploy=False):
        unseen = np.flatnonzero(self.n == 0)
        if len(unseen) and not deploy:
            return int(unseen[t % len(unseen)])
        return self.argmax([float(self.theta(a) @ x) for a in range(len(HARNESSES))])


POLICIES = (RandomPolicy, BestFixed, EpsilonGreedy, UCB1, Thompson, LinUCB,
            LinearThompson, SupervisedRewardModel)


def fallback_outcome(unit):
    direct = unit["outcomes"]["direct"]
    if direct["success"]:
        return dict(direct), ["direct"]
    verifier = unit["outcomes"]["verifier_loop"]
    # Conservative replay approximation: a failed direct call followed by a complete verifier-loop call.
    out = {"success": verifier["success"], "tokens": direct["tokens"] + verifier["tokens"],
           "tool_calls": direct["tool_calls"] + verifier["tool_calls"]}
    out["reward"] = out["success"] - .10 * out["tokens"] / 1000.0 - .02 * out["tool_calls"]
    return out, ["direct", "verifier_loop"]


def pools(units, scope):
    domains = {scope} if scope in {"python", "sql"} else {"python", "sql"}
    selected = [u for u in units if u["domain"] in domains]
    return ([u for u in selected if u["split"] in {"search", "online"}],
            [u for u in selected if u["split"] == "validation"])


def transfer_pools(units, scope):
    source, target = scope.split("_to_")
    src = [u for u in units if u["domain"] == source and u["split"] in {"search", "online"}]
    train = [u for u in units if u["domain"] == target and u["split"] in {"search", "online"}]
    validation = [u for u in units if u["domain"] == target and u["split"] == "validation"]
    return src, train, validation


def best_fixed_arm(train):
    scores = [mean([u["outcomes"][h]["reward"] for u in train]) for h in HARNESSES]
    return int(np.argmax(scores)), scores


def evaluate(pol, validation, encoder, rng, fallback=False):
    outcomes, choices, charged = [], [], 0
    for t, unit in enumerate(validation):
        if fallback:
            out, attempts = fallback_outcome(unit)
            choice = "->".join(attempts)
            charged += len(attempts)
        else:
            arm = pol.select(encoder(unit), t, deploy=True)
            choice = HARNESSES[arm]
            out = unit["outcomes"][choice]
            charged += 1
        outcomes.append(out)
        choices.append((unit["domain"], choice))
    return {
        "validation_reward": mean([x["reward"] for x in outcomes]),
        "validation_success": mean([x["success"] for x in outcomes]),
        "validation_tokens": mean([x["tokens"] for x in outcomes]),
        "validation_tool_calls": mean([x["tool_calls"] for x in outcomes]),
        "validation_task_units": len(validation), "validation_charged_evaluations": charged,
        "selection_share": dict(Counter(x[1] for x in choices)),
        "selection_by_domain": {d: dict(Counter(x[1] for x in choices if x[0] == d))
                                for d in sorted({x[0] for x in choices})},
    }


def train_policy(P, stream, budget, encoder, rng, comparator, log, meta, start_t=0):
    pol = P(rng, len(encoder(stream[0])) if stream else len(encoder.keys) + 1)
    cumulative_fixed = cumulative_oracle = 0.0
    for t, unit in enumerate(stream[:budget]):
        x = encoder(unit)
        arm = pol.select(x, start_t + t, deploy=False)
        out = unit["outcomes"][HARNESSES[arm]]
        pol.update(arm, out["reward"], x)
        cumulative_fixed += unit["outcomes"][HARNESSES[comparator]]["reward"] - out["reward"]
        cumulative_oracle += max(v["reward"] for v in unit["outcomes"].values()) - out["reward"]
        log.write(json.dumps({**meta, "phase": "target_replay", "t": t, "task_key": unit["key"],
                              "selected_harness": HARNESSES[arm], **out,
                              "cumulative_regret_vs_best_fixed": cumulative_fixed,
                              "cumulative_regret_vs_task_oracle": cumulative_oracle,
                              "policy_state": pol.state()}, ensure_ascii=False) + "\n")
    return pol, cumulative_fixed, cumulative_oracle


def pretrain(pol, stream, encoder, log, meta):
    for t, unit in enumerate(stream[:SOURCE_PRETRAIN_BUDGET]):
        x = encoder(unit)
        arm = pol.select(x, t, deploy=False)
        out = unit["outcomes"][HARNESSES[arm]]
        pol.update(arm, out["reward"], x)
        log.write(json.dumps({**meta, "phase": "source_pretrain", "t": t, "task_key": unit["key"],
                              "selected_harness": HARNESSES[arm], **out,
                              "policy_state": pol.state()}, ensure_ascii=False) + "\n")


def run_one(P, train, validation, encoder, perm, budget, seed, scope, log, source=None):
    rng = np.random.default_rng([BASE_SEED, seed, budget, POLICIES.index(P), len(scope)])
    comparator, _ = best_fixed_arm(train)
    meta = {"replay_seed": seed, "budget": budget, "scope": scope, "policy": P.name}
    pol = P(rng, len(encoder(train[0])) if train else len(encoder.keys) + 1)
    if source:
        pretrain(pol, source, encoder, log, meta)
    cumulative_fixed = cumulative_oracle = 0.0
    for t, unit in enumerate(perm[:budget]):
        x = encoder(unit)
        arm = pol.select(x, t + (SOURCE_PRETRAIN_BUDGET if source else 0), deploy=False)
        out = unit["outcomes"][HARNESSES[arm]]
        pol.update(arm, out["reward"], x)
        cumulative_fixed += unit["outcomes"][HARNESSES[comparator]]["reward"] - out["reward"]
        cumulative_oracle += max(v["reward"] for v in unit["outcomes"].values()) - out["reward"]
        log.write(json.dumps({**meta, "phase": "target_replay", "t": t, "task_key": unit["key"],
                              "selected_harness": HARNESSES[arm], **out,
                              "cumulative_regret_vs_best_fixed": cumulative_fixed,
                              "cumulative_regret_vs_task_oracle": cumulative_oracle,
                              "policy_state": pol.state()}, ensure_ascii=False) + "\n")
    pol.frozen = True
    ev = evaluate(pol, validation, encoder, rng)
    return {**meta, "source_pretrain_budget": SOURCE_PRETRAIN_BUDGET if source else 0,
            "cumulative_regret_vs_best_fixed": cumulative_fixed,
            "cumulative_regret_vs_task_oracle": cumulative_oracle, "final_state": pol.state(), **ev}


def run_fallback(validation, budget, seed, scope):
    ev = evaluate(None, validation, None, None, fallback=True)
    return {"replay_seed": seed, "budget": budget, "scope": scope,
            "policy": "direct_then_verifier_on_failure", "source_pretrain_budget": 0,
            "cumulative_regret_vs_best_fixed": None, "cumulative_regret_vs_task_oracle": None,
            "final_state": {"fixed_rule": True}, **ev}


def aggregate(results):
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["scope"], r["budget"], r["policy"])].append(r)
    out = []
    metrics = ("validation_reward", "validation_success", "validation_tokens", "validation_tool_calls",
               "cumulative_regret_vs_best_fixed", "cumulative_regret_vs_task_oracle")
    for (scope, budget, policy), rr in sorted(grouped.items()):
        row = {"scope": scope, "budget": budget, "policy": policy, "n_seeds": len(rr),
               "source_pretrain_budget": rr[0]["source_pretrain_budget"]}
        for m in metrics:
            vals = [x[m] for x in rr if x[m] is not None]
            row[m + "_ci95"] = ci95(vals)
        row["mean_validation_charged_evaluations"] = mean([x["validation_charged_evaluations"] for x in rr])
        out.append(row)
    return out


def pick_finalists(agg):
    candidates = {"epsilon_greedy", "ucb1", "thompson_sampling", "linucb", "linear_thompson_sampling"}
    rows = [r for r in agg if r["scope"] == "combined" and r["budget"] in {25, 50}
            and r["policy"] in candidates]
    base = {(r["budget"]): r for r in agg if r["scope"] == "combined"
            and r["policy"] == "best_fixed_round_robin" and r["budget"] in {25, 50}}
    by = defaultdict(list)
    for r in rows:
        b = base[r["budget"]]
        effect = r["validation_reward_ci95"][0] - b["validation_reward_ci95"][0]
        sdrop = b["validation_success_ci95"][0] - r["validation_success_ci95"][0]
        by[r["policy"]].append((effect, sdrop, r))
    ranked = []
    for policy, vals in by.items():
        ranked.append({"policy": policy, "mean_reward_effect_vs_best_fixed": mean([x[0] for x in vals]),
                       "mean_success_drop_vs_best_fixed": mean([x[1] for x in vals]),
                       "qualifies_registered_gate": all(x[0] >= .02 and x[1] <= .01 for x in vals)})
    ranked.sort(key=lambda x: (x["qualifies_registered_gate"], x["mean_reward_effect_vs_best_fixed"],
                               -x["mean_success_drop_vs_best_fixed"]), reverse=True)
    return ranked[:2], ranked


def render_report(summary):
    lines = ["# Сравнение малоданных стратегий выбора harness", "",
             "Replay — проигрывание по заранее собранной матрице — выполнен только на non-heldout данных. "
             "Validation — проверочный набор — нигде не обновляет policy (стратегию выбора).", "",
             "Основная награда: `success - 0.10 × tokens/1000 - 0.02 × tool_calls`; latency в неё не входит.", "",
             "## Результаты validation", "",
             "| область | бюджет | метод | reward (95% CI) | success (95% CI) | regret к fixed |", "|---|---:|---|---:|---:|---:|"]
    for r in summary["aggregate"]:
        vr, vs, rg = r["validation_reward_ci95"], r["validation_success_ci95"], r["cumulative_regret_vs_best_fixed_ci95"]
        fmt = lambda x: "—" if x[0] is None else f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"
        lines.append(f"| {r['scope']} | {r['budget']} | `{r['policy']}` | {fmt(vr)} | {fmt(vs)} | {fmt(rg)} |")
    lines += ["", "## Замороженные финалисты", ""]
    for x in summary["finalists"]:
        lines.append(f"- `{x['policy']}`: эффект reward к честному best-fixed {x['mean_reward_effect_vs_best_fixed']:+.3f}; "
                     f"изменение success (drop) {x['mean_success_drop_vs_best_fixed']:+.3f}; "
                     f"предзарегистрированный gate: {'пройден' if x['qualifies_registered_gate'] else 'не пройден'}.")
    lines += ["", "Фиксированное правило `direct → verifier_loop` при неуспехе является консервативной replay-аппроксимацией: "
              "при эскалации ему начислена стоимость обоих полных запусков. Поэтому число charged evaluations на validation может превышать число задач.", "",
              "Перенос `python_to_sql` и `sql_to_python` использует 50 bandit-наблюдений исходного домена и затем 0/5/10/25/50 наблюдений целевого домена; исходные labels не раскрывают контрфактические arms.", "",
              "## Этап 2", "",
              "Конфигурации не более двух финалистов заморожены в `results/policy/frozen_finalists/`. Реальный новый LM Studio поток не был подменён replay: ни один финалист не прошёл предзарегистрированный gate, а нового task manifest нет. `online_log.jsonl.gz` содержит только явную status-запись без task labels; подробные события replay находятся в `replay_log.jsonl.gz`.", "",
              "## Воспроизведение", "", "```powershell",
              "python pilot/policy_selection.py", "python -m pytest pilot/tests/test_policy_selection.py -q", "```", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results" / "policy")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "policy_comparison.md")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if not a.output_dir.is_absolute():
        a.output_dir = ROOT / a.output_dir
    if not a.report.is_absolute():
        a.report = ROOT / a.report
    gate = json.loads((ROOT / "reports" / "go_no_go.json").read_text(encoding="utf-8"))
    if gate.get("decision") == "NO_SIGNAL":
        raise SystemExit("go_no_go.json says NO_SIGNAL; RL/replay is forbidden")
    expected = [a.output_dir / "online_log.jsonl.gz", a.output_dir / "replay_log.jsonl.gz",
                a.output_dir / "per_seed.jsonl",
                a.output_dir / "summary.json", a.output_dir / "integrity_report.json", a.report]
    if any(p.exists() for p in expected) and not a.overwrite:
        raise SystemExit("refusing to overwrite policy results; pass --overwrite explicitly")
    if a.output_dir.exists() and a.overwrite:
        shutil.rmtree(a.output_dir)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    units, integrity = load_units()
    scopes = ("python", "sql", "combined", "python_to_sql", "sql_to_python")
    results = []
    with gzip.open(a.output_dir / "replay_log.jsonl.gz", "wt", encoding="utf-8") as log:
        for scope in scopes:
            if "_to_" in scope:
                source, train, validation = transfer_pools(units, scope)
            else:
                source = None
                train, validation = pools(units, scope)
            encoder = Encoder((source or []) + train + validation, cross_domain=(scope == "combined"))
            for seed in range(a.seeds):
                rng = random.Random([a.base_seed, seed, len(scope)].__repr__())
                perm = list(train)
                rng.shuffle(perm)
                src_perm = list(source or [])
                rng.shuffle(src_perm)
                for budget in BUDGETS:
                    if budget > len(perm):
                        raise ValueError(f"scope {scope} has only {len(perm)} train units for budget {budget}")
                    for P in POLICIES:
                        results.append(run_one(P, train, validation, encoder, perm, budget, seed, scope, log,
                                               source=src_perm if source is not None else None))
                    results.append(run_fallback(validation, budget, seed, scope))
    agg = aggregate(results)
    finalists, ranking = pick_finalists(agg)
    frozen = a.output_dir / "frozen_finalists"
    frozen.mkdir()
    for rank, x in enumerate(finalists, 1):
        cfg = {**x, "rank": rank, "frozen_at_stage": "post-replay-pre-online",
               "selection_scope": "combined", "selection_budgets": [25, 50],
               "reward": {"lambda_tokens": .10, "lambda_tools": .02, "latency_in_reward": False},
               "harnesses": list(HARNESSES), "target_model": integrity["configs"]["python"]["model"],
               "temperature": 0.0, "features_available_pre_decision_only": True}
        (frozen / f"{rank:02d}_{x['policy']}.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"decision_gate": gate, "seeds": a.seeds, "base_seed": a.base_seed,
               "budgets": list(BUDGETS), "harnesses": list(HARNESSES), "reward": {
                   "lambda_tokens": .10, "lambda_tools": .02, "latency_in_reward": False},
               "split_policy": {"updates": ["search", "online"], "validation": "score-only", "heldout": "unread"},
               "source_pretrain_budget": SOURCE_PRETRAIN_BUDGET, "aggregate": agg,
               "finalists": finalists, "full_finalist_ranking": ranking,
               "stage2_status": ("pending_new_frozen_online_manifest_and_live_lm_studio"
                                  if any(x["qualifies_registered_gate"] for x in finalists)
                                  else "not_run_registered_gate_failed_and_no_new_online_manifest")}
    (a.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with gzip.open(a.output_dir / "online_log.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "online", "status": summary["stage2_status"],
                            "task_labels_observed": 0,
                            "reason": "no finalist passed the registered gate and no new frozen online manifest exists",
                            "frozen_finalists": [x["policy"] for x in finalists]}, ensure_ascii=False) + "\n")
    with (a.output_dir / "per_seed.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    integrity.update({"go_no_go_decision": gate["decision"], "heldout_rows_read": 0,
                      "validation_updates": 0, "policy_state_logged_after_every_update": True,
                      "same_target_stream_within_seed_budget_scope": True,
                      "explicit_replay_seeds": a.seeds, "source_pretrain_budget": SOURCE_PRETRAIN_BUDGET,
                      "outputs": {str(p.relative_to(ROOT)): file_hash(p) for p in
                                  [a.output_dir / "online_log.jsonl.gz", a.output_dir / "replay_log.jsonl.gz",
                                   a.output_dir / "per_seed.jsonl",
                                   a.output_dir / "summary.json"]}})
    (a.output_dir / "integrity_report.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")
    a.report.write_text(render_report(summary), encoding="utf-8")
    print(f"wrote {a.report}")
    print("finalists:", ", ".join(x["policy"] for x in finalists))
    return summary


if __name__ == "__main__":
    main()
