"""Stage 2: online bandit learning over the cached outcome table + frozen held-out evaluation.

For every seed: one random permutation of the 100-task online pool; budget T in {10,25,50} uses its first T
tasks (identical stream and identical total task budget for every policy). Each online step = one task run
with one pattern = one observation. After T steps the policy is frozen and deploys its greedy choice on the
50 held-out tasks; held-out outcomes are only read for scoring, never passed to update().

  python bandit/simulate.py --seeds 50 --budgets 10,25,50 --lam-tok 0.1 --lam-tool 0.02 --tag main
"""

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from patterns import PATTERNS  # noqa: E402
from policies import EXTRA, REQUIRED  # noqa: E402

CACHE = HERE / "cache"


def load_table(cache=CACHE):
    rows = [json.loads(x) for x in (Path(cache) / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()]
    cfg = json.loads((Path(cache) / "config.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if r["harness_hash"] == cfg["harness_hash"]]
    tab = defaultdict(dict)
    for r in rows:
        tab[r["task_id"]][r["pattern"]] = r
    pools = {}
    for p in ("online", "heldout"):
        ids = [i for i in cfg["pools"][p] if len(tab.get(i, {})) == len(PATTERNS)]
        if len(ids) < len(cfg["pools"][p]):
            print(f"WARNING: {p}: {len(cfg['pools'][p]) - len(ids)} tasks incomplete in cache, skipped")
        pools[p] = ids
    types = {int(k): v for k, v in cfg["task_types"].items()}
    return tab, pools, types, cfg


class Reward:
    def __init__(self, lam_tok, lam_tool, lam_lat):
        self.lam_tok, self.lam_tool, self.lam_lat = lam_tok, lam_tool, lam_lat

    def __call__(self, row):
        return (row["success"] - self.lam_tok * row["tokens"] / 1000 - self.lam_tool * row["tool_calls"]
                - self.lam_lat * row["latency_s"])

    def bounds(self, tab):
        v = [self(r) for t in tab.values() for r in t.values()]
        return min(min(v), 0.0), max(max(v), 1.0)

    def as_dict(self):
        return {"lam_tok_per_1k": self.lam_tok, "lam_tool": self.lam_tool, "lam_latency_per_s": self.lam_lat}


def ci95(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def run(a):
    tab, pools, types, cfg = load_table(a.cache)
    R = Reward(a.lam_tok, a.lam_tool, a.lam_lat)
    r_lo, r_hi = R.bounds(tab)  # fixed a-priori range; only used to rescale rewards to [0,1]
    K = len(PATTERNS)
    online, heldout = pools["online"], pools["heldout"]
    online_set = set(online)
    rew = {i: np.array([R(tab[i][p]) for p in PATTERNS]) for i in online + heldout}
    mu_online = np.mean([rew[i] for i in online], axis=0)  # full-information arm means (regret reference only)
    a_star = int(np.argmax(mu_online))
    budgets = [int(b) for b in a.budgets.split(",")]
    policies = REQUIRED + (EXTRA if a.extra else [])
    out = HERE / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    log = gzip.open(out / "online_log.jsonl.gz", "wt", encoding="utf-8")
    res = []
    for seed in range(a.seeds):
        stream_all = list(np.random.default_rng([a.base_seed, seed]).permutation(online))
        for T in budgets:
            stream = stream_all[:T]
            for pi, P in enumerate(policies):
                rng = np.random.default_rng([a.base_seed, seed, T, pi])
                pol = P(PATTERNS, rng, r_lo=r_lo, r_hi=r_hi)
                cum_fixed = cum_oracle = 0.0
                tot = Counter()
                picks = []
                for t, x in enumerate(stream):
                    assert x in online_set
                    ctx = types[x]
                    arm = pol.select(t, ctx)
                    row = tab[x][PATTERNS[arm]]
                    r = R(row)
                    pol.update(arm, r, ctx)
                    cum_fixed += mu_online[a_star] - mu_online[arm]
                    cum_oracle += rew[x].max() - r
                    tot.update(success=row["success"], tokens=row["tokens"], tool_calls=row["tool_calls"])
                    picks.append((ctx, PATTERNS[arm]))
                    log.write(json.dumps({
                        "seed": seed, "budget": T, "policy": pol.name, "t": t, "task_id": int(x), "type": ctx,
                        "pattern": PATTERNS[arm], "reward": round(r, 5), "success": row["success"],
                        "tokens": row["tokens"], "tool_calls": row["tool_calls"], "latency_s": row["latency_s"],
                        "regret_vs_best_fixed": round(cum_fixed, 5), "regret_vs_task_oracle": round(cum_oracle, 5),
                        "state": pol.state()}) + "\n")
                pol.frozen = True  # from here on any update() raises
                h = Counter()
                dep = []
                for x in heldout:
                    arm = pol.deploy(types[x])
                    row = tab[x][PATTERNS[arm]]
                    h.update(reward=R(row), success=row["success"], tokens=row["tokens"],
                             tool_calls=row["tool_calls"])
                    dep.append(PATTERNS[arm])
                n = len(heldout)
                res.append({
                    "seed": seed, "budget": T, "policy": pol.name,
                    "cum_regret_vs_best_fixed": cum_fixed, "cum_regret_vs_task_oracle": cum_oracle,
                    "online_mean_reward": sum(rew[x][PATTERNS.index(p)] for x, (_, p) in zip(stream, picks)) / T,
                    "online_success": tot["success"] / T, "online_tokens": tot["tokens"] / T,
                    "heldout_reward": h["reward"] / n, "heldout_success": h["success"] / n,
                    "heldout_tokens": h["tokens"] / n, "heldout_tool_calls": h["tool_calls"] / n,
                    "deployed": Counter(dep).most_common(1)[0][0] if pol.name != "random" else "random",
                    "picks": picks, "final_state": pol.state()})
    log.close()

    # --------------------------------------------------------------------------- references (full information)
    def fixed_heldout(arm):
        return {"reward": float(np.mean([rew[x][arm] for x in heldout])),
                "success": float(np.mean([tab[x][PATTERNS[arm]]["success"] for x in heldout])),
                "tokens": float(np.mean([tab[x][PATTERNS[arm]]["tokens"] for x in heldout])),
                "tool_calls": float(np.mean([tab[x][PATTERNS[arm]]["tool_calls"] for x in heldout]))}

    per_pattern = {}
    for k, p in enumerate(PATTERNS):
        on = [tab[x][p] for x in online]
        per_pattern[p] = {
            "online_reward": float(mu_online[k]), "online_success": float(np.mean([r["success"] for r in on])),
            "online_tokens": float(np.mean([r["tokens"] for r in on])),
            "online_tool_calls": float(np.mean([r["tool_calls"] for r in on])),
            "online_lm_calls": float(np.mean([r["lm_calls"] for r in on])),
            **{f"heldout_{kk}": v for kk, v in fixed_heldout(k).items()}}
    type_names = sorted({types[x] for x in online + heldout})
    by_type = {}
    for pool, ids in (("online", online), ("heldout", heldout)):
        for ty in type_names:
            sub = [x for x in ids if types[x] == ty]
            if sub:
                by_type[f"{pool}/{ty}"] = {"n": len(sub), **{
                    p: {"reward": float(np.mean([rew[x][k] for x in sub])),
                        "success": float(np.mean([tab[x][p]["success"] for x in sub]))}
                    for k, p in enumerate(PATTERNS)}}
    # per-type oracle: best pattern per type chosen on the ONLINE pool (full info), evaluated on held-out
    best_type_arm = {ty: int(np.argmax([by_type[f"online/{ty}"][p]["reward"] for p in PATTERNS]))
                     for ty in type_names if f"online/{ty}" in by_type}
    per_type_ref = [rew[x][best_type_arm.get(types[x], a_star)] for x in heldout]
    references = {
        "best_fixed_oracle(online-full-info)": {"pattern": PATTERNS[a_star], **fixed_heldout(a_star)},
        "per_type_oracle(online-full-info)": {"patterns": {t: PATTERNS[k] for t, k in best_type_arm.items()},
                                              "reward": float(np.mean(per_type_ref))},
        "per_task_oracle(heldout, upper bound)": {"reward": float(np.mean([rew[x].max() for x in heldout])),
                                                  "success": float(np.mean([max(tab[x][p]["success"]
                                                                                for p in PATTERNS)
                                                                            for x in heldout]))},
        "random_expected": {"reward": float(np.mean([rew[x].mean() for x in heldout]))},
    }

    # --------------------------------------------------------------------------- aggregate
    agg = []
    for T in budgets:
        for P in policies:
            rs = [r for r in res if r["budget"] == T and r["policy"] == P.name]
            row = {"budget": T, "policy": P.name, "n_seeds": len(rs)}
            for key in ("cum_regret_vs_best_fixed", "cum_regret_vs_task_oracle", "online_mean_reward",
                        "online_success", "online_tokens", "heldout_reward", "heldout_success", "heldout_tokens",
                        "heldout_tool_calls"):
                row[key], row[key + "_ci95"] = ci95([r[key] for r in rs])
            row["deployed_dist"] = dict(Counter(r["deployed"] for r in rs).most_common())
            row["p_deploy_best"] = float(np.mean([r["deployed"] == PATTERNS[a_star] for r in rs]))
            sel = Counter(p for r in rs for _, p in r["picks"])
            row["online_selection_share"] = {p: sel[p] / (T * len(rs)) for p in PATTERNS}
            # does the (non-contextual) policy pick different patterns for different task types?
            by_ty = defaultdict(Counter)
            for r in rs:
                for ty, p in r["picks"]:
                    by_ty[ty][p] += 1
            row["selection_share_by_type"] = {ty: {p: c[p] / sum(c.values()) for p in PATTERNS}
                                              for ty, c in sorted(by_ty.items())}
            agg.append(row)

    # latency reliability: verifier_loop that passes the visible test first try sends the identical request
    # as direct (same messages, same tokens) -> any latency gap is pure measurement noise.
    pairs = [(tab[x]["direct"]["latency_s"], tab[x]["verifier_loop"]["latency_s"]) for x in online + heldout
             if tab[x]["verifier_loop"]["lm_calls"] == 1
             and tab[x]["verifier_loop"]["tokens"] == tab[x]["direct"]["tokens"]]
    lat = {}
    if pairs:
        d, v = np.array(pairs).T
        rel = np.abs(d - v) / np.maximum((d + v) / 2, 1e-9)
        all_rows = [r for t in tab.values() for r in t.values()]
        lat = {"n_identical_request_pairs": len(pairs),
               "median_rel_abs_diff": float(np.median(rel)), "p90_rel_abs_diff": float(np.quantile(rel, 0.9)),
               "corr_latency_tokens": float(np.corrcoef([r["latency_s"] for r in all_rows],
                                                        [r["tokens"] for r in all_rows])[0, 1])}

    summary = {"tag": a.tag, "reward": R.as_dict(), "reward_range_for_rescaling": [r_lo, r_hi],
               "seeds": a.seeds, "base_seed": a.base_seed, "budgets": budgets, "patterns": PATTERNS,
               "n_online": len(online), "n_heldout": len(heldout), "harness_hash": cfg["harness_hash"],
               "target_model": cfg["target_model"], "target_options": cfg["target_options"],
               "best_fixed_arm_online": PATTERNS[a_star], "per_pattern": per_pattern, "by_type": by_type,
               "references": references, "aggregate": agg, "latency_reliability": lat, "argv": sys.argv}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with open(out / "per_seed.jsonl", "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps({k: v for k, v in r.items() if k != "picks"}) + "\n")
    (out / "report.md").write_text(report(summary), encoding="utf-8")
    sys.stdout.buffer.write(report(summary).encode("utf-8"))
    return summary


def report(s):
    P = s["patterns"]
    L = [f"## Run `{s['tag']}`: reward = success − {s['reward']['lam_tok_per_1k']}·tokens/1000 − "
         f"{s['reward']['lam_tool']}·tool_calls − {s['reward']['lam_latency_per_s']}·latency_s; "
         f"{s['seeds']} seeds; online pool {s['n_online']}, held-out {s['n_heldout']}\n",
         "### Patterns (full information, each pattern on every task)\n",
         "| pattern | online reward | online success | held-out reward | held-out success | tokens/task | "
         "LM calls/task | tool calls/task |", "|---|---|---|---|---|---|---|---|"]
    for p in P:
        x = s["per_pattern"][p]
        L.append(f"| {p} | {x['online_reward']:.3f} | {x['online_success']:.2f} | {x['heldout_reward']:.3f} | "
                 f"{x['heldout_success']:.2f} | {x['online_tokens']:.0f} | {x['online_lm_calls']:.2f} | "
                 f"{x['online_tool_calls']:.2f} |")
    ref = s["references"]
    L += ["", f"Best fixed pattern on the online pool (regret comparator): **{s['best_fixed_arm_online']}**.", "",
          "### Online cumulative regret and held-out quality (mean ± 95% CI over seeds)\n",
          "| T | policy | regret vs best fixed | regret vs per-task oracle | online success | held-out reward | "
          "held-out success | held-out tokens | P(deploy best) | deployed |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["aggregate"]:
        dep = ", ".join(f"{k}:{v}" for k, v in r["deployed_dist"].items())
        L.append(f"| {r['budget']} | {r['policy']} | {r['cum_regret_vs_best_fixed']:.2f} ± "
                 f"{r['cum_regret_vs_best_fixed_ci95']:.2f} | {r['cum_regret_vs_task_oracle']:.2f} ± "
                 f"{r['cum_regret_vs_task_oracle_ci95']:.2f} | {r['online_success']:.3f} | "
                 f"{r['heldout_reward']:.3f} ± {r['heldout_reward_ci95']:.3f} | {r['heldout_success']:.3f} | "
                 f"{r['heldout_tokens']:.0f} | {r['p_deploy_best']:.2f} | {dep} |")
    L += ["", "References on held-out: "
          f"best fixed (full info) = {ref['best_fixed_oracle(online-full-info)']['pattern']} "
          f"{ref['best_fixed_oracle(online-full-info)']['reward']:.3f}; "
          f"per-type oracle {ref['per_type_oracle(online-full-info)']['reward']:.3f} "
          f"({ref['per_type_oracle(online-full-info)']['patterns']}); "
          f"per-task oracle {ref['per_task_oracle(heldout, upper bound)']['reward']:.3f} "
          f"(success {ref['per_task_oracle(heldout, upper bound)']['success']:.2f}); "
          f"random expected {ref['random_expected']['reward']:.3f}.", "",
          "### Mean reward by task type × pattern (full information)\n",
          "| pool/type | n | " + " | ".join(P) + " | best |", "|---|---|" + "---|" * (len(P) + 1)]
    for k, v in s["by_type"].items():
        vals = [v[p]["reward"] for p in P]
        L.append(f"| {k} | {v['n']} | " + " | ".join(f"{x:.3f} ({v[p]['success']:.2f})" for x, p in zip(vals, P))
                 + f" | {P[int(np.argmax(vals))]} |")
    L += ["", "### Online selection share by task type, T=50\n",
          "| policy | type | " + " | ".join(P) + " |", "|---|---|" + "---|" * len(P)]
    for r in s["aggregate"]:
        if r["budget"] != max(s["budgets"]):
            continue
        for ty, sh in r["selection_share_by_type"].items():
            L.append(f"| {r['policy']} | {ty} | " + " | ".join(f"{sh[p]:.2f}" for p in P) + " |")
    if s["latency_reliability"]:
        lr = s["latency_reliability"]
        L += ["", f"Latency reliability: {lr['n_identical_request_pairs']} identical-request pairs "
              f"(direct vs verifier_loop passing first try): median relative |Δlatency| = "
              f"{lr['median_rel_abs_diff']:.2f}, p90 = {lr['p90_rel_abs_diff']:.2f}; "
              f"corr(latency, tokens) = {lr['corr_latency_tokens']:.2f}."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--base-seed", type=int, default=2026)
    ap.add_argument("--budgets", default="10,25,50")
    ap.add_argument("--lam-tok", type=float, default=0.1, help="penalty per 1000 tokens")
    ap.add_argument("--lam-tool", type=float, default=0.02, help="penalty per tool call")
    ap.add_argument("--lam-lat", type=float, default=0.0, help="penalty per second of LM latency")
    ap.add_argument("--no-extra", dest="extra", action="store_false", help="skip the per-type TS extension")
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--tag", default="main")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
