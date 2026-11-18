"""Command-line entry point for smoke, complete core, resume, and audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import write_reports
from .engine import METHODS, ROOT, ExperimentRun, create_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--run-id", required=True)
        cmd.add_argument("--protocol", type=Path, default=ROOT / "experiments" / "harness_optimization" / "preregistered_protocol.json")
    audit_cmd = sub.add_parser("audit")
    audit_cmd.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "audit":
        summary, integrity = write_reports(args.run_dir.resolve())
        print(json.dumps({"status": summary["status"], "best_method": summary["best_method"],
                          "audit": integrity["verdict"]}, ensure_ascii=False, indent=2))
        return

    run_dir = ROOT / "results" / "harness_optimization" / args.run_id
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    create_manifest(run_dir, protocol_path, args.run_id, args.command)
    experiment = ExperimentRun(run_dir, protocol)
    if args.command == "smoke":
        experiment.run_matrix(methods=METHODS, seeds=(0,), n_tasks=3, n_candidates=2, validation_tasks=3)
    else:
        experiment.run_matrix()
    summary, integrity = write_reports(run_dir)
    print(json.dumps({"run_dir": str(run_dir), "status": summary["status"],
                      "best_method": summary["best_method"], "audit": integrity["verdict"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

