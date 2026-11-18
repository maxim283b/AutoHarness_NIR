"""Create the small deterministic JSONL fixture used for the example report."""
import json
from pathlib import Path


def rows():
    out = []
    for i in range(24):
        for seed in (0, 1):
            for harness in ("direct", "verifier_loop"):
                best_direct = i % 2 == 0
                out.append({
                    "task_id": f"task-{i}", "task_template": f"template-{i % 6}", "split": "search",
                    "seed": seed, "harness": harness, "model": "synthetic@sha256:test",
                    "success": int((harness == "direct") == best_direct),
                    "tokens": 100 if harness == "direct" else 200,
                    "tool_calls": 0 if harness == "direct" else 1, "latency_s": 1.0,
                    "task_features": {"even": int(i % 2 == 0), "index": i},
                    "trajectory": [{"step": "solve", "input_hash": "x", "output_hash": "y"}],
                })
    # Held-out is included only to demonstrate score-only reporting.
    out.append({**out[0], "task_id": "heldout-1", "split": "heldout"})
    return out


if __name__ == "__main__":
    target = Path(__file__).parent / "data"
    target.mkdir(exist_ok=True)
    (target / "synthetic.jsonl").write_text("\n".join(json.dumps(r) for r in rows()) + "\n", encoding="utf-8")
