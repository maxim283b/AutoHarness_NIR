from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline_adapter import write_probe


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("probe-baseline")
    sub.add_parser("run-confirmatory")
    args = parser.parse_args()
    if args.command == "probe-baseline":
        path = ROOT / "results" / "final_harness_experiment" / "gate_probe_20261012" / "baseline_probe.json"
        print(json.dumps(write_probe(ROOT, path), ensure_ascii=False, indent=2))
        return
    protocol = json.loads((ROOT / "experiments" / "final_harness_experiment" / "preregistered_protocol.json").read_text(encoding="utf-8"))
    if not protocol["gate_a"]["confirmatory_run_allowed"]:
        raise SystemExit("CONFIRMATORY_RUN_BLOCKED: Gate A has status blocked_no_reproducible_open_baseline")


if __name__ == "__main__":
    main()
