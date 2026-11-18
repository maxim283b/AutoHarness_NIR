"""Stage 1: run every pattern on every online + held-out task once with the frozen target (temperature 0).

The result is a full-information outcome table. Because the target is deterministic (temperature=0, fixed
seed) and every pattern is a pure function of the task (the memory store is static), "bandit pulls arm a on
task x" during online learning returns exactly the cached row (x, a). Stage 2 (simulate.py) therefore replays
the table instead of re-querying the LLM for every policy x seed x budget, and the same table gives exact
counterfactuals for regret. Resumable: rows already in outcomes.jsonl are skipped.

  python bandit/collect.py --workers 2
"""

import argparse
import json
import platform
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from harness import TARGET_MODEL, TARGET_OPTIONS, sha  # noqa: E402
from llm import TokenLedger, make_chat_client  # noqa: E402
from patterns import FUNCS, PATTERNS, MemoryBank, run_pattern  # noqa: E402
from tasks import build_pools  # noqa: E402

import inspect  # noqa: E402
import patterns as _patterns_mod  # noqa: E402

OUT = HERE / "cache"


def harness_hash() -> str:
    return sha(inspect.getsource(_patterns_mod))[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pools", default="online,heldout")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N tasks per pool")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pools = build_pools()
    mem = MemoryBank(pools["memory"])
    hh = harness_hash()
    cfg = {"target_model": TARGET_MODEL, "target_options": TARGET_OPTIONS, "patterns": PATTERNS,
           "harness_hash": hh, "pools": {k: [t["task_id"] for t in v] for k, v in pools.items()},
           "task_types": {str(t["task_id"]): t["type"] for p in ("online", "heldout") for t in pools[p]},
           "platform": platform.platform(), "python": sys.version, "argv": sys.argv}
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    rows_path, traj_path = out / "outcomes.jsonl", out / "trajectories.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["harness_hash"] == hh:
                done.add((r["task_id"], r["pattern"]))
    jobs = [(p, t, pat) for p in a.pools.split(",") for t in pools[p][: a.limit] for pat in PATTERNS
            if (t["task_id"], pat) not in done]
    print(f"harness {hh}: {len(done)} cached, {len(jobs)} to run", flush=True)

    ledger = TokenLedger()
    client = make_chat_client(TARGET_MODEL, TARGET_OPTIONS, ledger, "target")
    lock = threading.Lock()
    t0 = time.time()

    def work(job):
        pool, task, pat = job
        return pool, task, run_pattern(pat, task, client, mem)

    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            pool, task, r = f.result()
            steps = r.pop("steps")
            row = {**r, "pool": pool, "type": task["type"], "harness_hash": hh, "workers": a.workers}
            with lock:
                with open(rows_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                with open(traj_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"task_id": r["task_id"], "pattern": r["pattern"], "harness_hash": hh,
                                         "steps": steps}, ensure_ascii=False) + "\n")
            el = time.time() - t0
            print(f"[{i + 1}/{len(jobs)}] {pool} task {r['task_id']:>4} {r['pattern']:<13} ok={r['success']} "
                  f"tok={r['tokens']:>5} tools={r['tool_calls']} lat={r['latency_s']:.1f}s  "
                  f"eta={el / (i + 1) * (len(jobs) - i - 1) / 60:.0f}min", flush=True)
    print("ledger", json.dumps(ledger.snapshot()), flush=True)


if __name__ == "__main__":
    main()
