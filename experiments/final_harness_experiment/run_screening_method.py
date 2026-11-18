"""Run or resume one six-candidate staged-screening method."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTOSADDLER_SRC = ROOT / "third_party/AutoSaddler/src"
for item in (AUTOSADDLER_SRC, ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from autosaddler.v2.config.registry import ScenarioRegistration, build_runtime, default_registry
from autosaddler.v2.core.policies import DevelopmentDecision
from autosaddler.v2.plugins.api import SCENARIO_PLUGIN_API_VERSION, ScenarioPlugin

from .screening_controller import METHODS, ScreeningController
from .sql_autosaddler_scenario import build_sql_harness_components


class FullAlwaysDevelopment:
    """Shared screening policy: every child gets the same development evaluation."""

    def decide(self, verdict):
        return DevelopmentDecision(evaluate=True, reason="staged screening evaluates every completed child")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_id = f"{args.method}-seed{args.seed}"
    run_root = ROOT / "results/final_harness_experiment/staged-screening-v1/method_runs"
    run_dir = run_root / run_id
    controller = ScreeningController(method=args.method, seed=args.seed, run_dir=run_dir)

    def scenario_factory(**kwargs):
        scenario = build_sql_harness_components(**kwargs)
        return replace(scenario, prompt_pack=controller.wrap_prompt_pack(scenario.prompt_pack))

    registry = default_registry()
    registry.register_scenario(
        ScenarioRegistration(
            plugin=ScenarioPlugin(name="sql_harness", api_version=SCENARIO_PLUGIN_API_VERSION, factory=scenario_factory),
            source="builtin",
        )
    )
    registry.development["full_always"] = FullAlwaysDevelopment
    runtime = build_runtime(args.config, run_id=run_id, registry=registry, transition_hook=controller.transition_hook)
    controller.reconcile(runtime.store.events())
    started = time.monotonic()
    result = runtime.engine.run()
    elapsed = time.monotonic() - started
    controller.reconcile(runtime.store.events())
    summary = {
        "run_id": run_id,
        "method": args.method,
        "seed": args.seed,
        "selected_candidate_id": result.selected_candidate_id,
        "development_score": result.development_score,
        "candidate_count": len(controller.state["history"]),
        "elapsed_seconds_this_invocation": elapsed,
        "policy_state_path": str(controller.path.relative_to(ROOT)),
    }
    (run_dir / "screening_method_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
