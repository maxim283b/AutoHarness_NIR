"""Compare per-task outcomes of two runs on the same tasks (reproducibility / paired check).

    python compare_runs.py runs/baseline runs/baseline_rep
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(run: Path) -> dict[int, dict]:
    rows = [json.loads(l) for l in (run / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["task_id"]: r for r in rows}


def main(a: str, b: str) -> None:
    ra, rb = load(Path(a)), load(Path(b))
    common = sorted(set(ra) & set(rb))
    score_diff = [t for t in common if ra[t]["score"] != rb[t]["score"]]
    answer_diff = [t for t in common if ra[t]["answer"] != rb[t]["answer"]]
    raw_diff = [t for t in common
                if [s["raw_response"] for s in ra[t]["trajectory"]] != [s["raw_response"] for s in rb[t]["trajectory"]]]
    out = {
        "a": a, "b": b, "n_common": len(common),
        "pass_a": sum(ra[t]["score"] for t in common), "pass_b": sum(rb[t]["score"] for t in common),
        "outcome_mismatches": len(score_diff), "outcome_mismatch_task_ids": score_diff,
        "answer_mismatches": len(answer_diff), "raw_response_mismatches": len(raw_diff),
        "a_only_pass": [t for t in score_diff if ra[t]["score"] > rb[t]["score"]],
        "b_only_pass": [t for t in score_diff if rb[t]["score"] > ra[t]["score"]],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
