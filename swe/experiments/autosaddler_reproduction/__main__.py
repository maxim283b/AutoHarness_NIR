import argparse
import json
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Shared OpenCode + AutoSaddler experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("smoke")
    for name in ("optimize", "evaluate", "calibrate"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--run-id", required=True)
        if name == "evaluate":
            p.add_argument("--test-repetitions", type=int, default=3)
            p.add_argument("--n-test", type=int)
    args = parser.parse_args()
    if args.command == "doctor":
        from .doctor import main as doctor
        return doctor()
    from .engine import build
    if args.command == "smoke":
        from .engine import prepare_seed
        _, revision = prepare_seed()
        implementation = hashlib.sha256(b"".join(p.name.encode()+p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()
        runtime = build(Path(__file__).with_name("configs") / "smoke.yaml", "s-" + revision[:6] + "-" + implementation[:6])
        result = runtime.engine.run()
        print(result)
        before = len(runtime.store.events())
        resumed = runtime.engine.run()
        assert resumed.selected_candidate_id == result.selected_candidate_id
        assert len(runtime.store.events()) == before, "Resume generated duplicate work"
        report = {"passed": True, "synthetic_only": True, "run_id": runtime.store.run_id, "selected_candidate_id": result.selected_candidate_id, "iterations": result.iterations, "resume_idempotent": True}
        root = Path(__file__).resolve().parents[2]
        (root / "results/autosaddler_reproduction/engine_smoke_result.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print("Resume is idempotent; these are synthetic checks, not benchmark scores")
        return 0
    runtime = build(args.config, args.run_id)
    if args.command == "optimize":
        from .doctor import check_real_configuration
        check_real_configuration(runtime)
        print(runtime.engine.run())
    elif args.command == "evaluate":
        from .runner import evaluate_final
        print(json.dumps(evaluate_final(runtime, args.test_repetitions, args.n_test), indent=2))
    else:
        from .doctor import check_real_configuration
        from .runner import run_case
        check_real_configuration(runtime)
        seed = runtime.scenario.harness_space.seed()
        materialized = runtime.scenario.harness_space.materialize(seed, "evaluate")
        try:
            chosen = {}
            for case in runtime.scenario.train_cases:
                chosen.setdefault(case.payload["benchmark"], case)
            result = {name: run_case(case, materialized.root, runtime.store.run_dir / "calibration" / name, runtime.config.scenario.settings, args.run_id, "target_search") for name, case in chosen.items()}
            (runtime.store.run_dir / "calibration.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
            print("Train-only calibration finished. Use measured token/quota usage to freeze the final experiment budget.")
        finally:
            materialized.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
