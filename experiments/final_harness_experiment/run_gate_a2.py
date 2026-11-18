"""Run one real-target candidate through the official AutoSaddler V2 engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTOSADDLER_SRC = ROOT / "third_party/AutoSaddler/src"
if str(AUTOSADDLER_SRC) not in sys.path:
    sys.path.insert(0, str(AUTOSADDLER_SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autosaddler.v2.config.registry import ScenarioRegistration, build_runtime, default_registry
from autosaddler.v2.plugins.api import SCENARIO_PLUGIN_API_VERSION, ScenarioPlugin

from experiments.final_harness_experiment.sql_autosaddler_scenario import build_sql_harness_components


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    registry = default_registry()
    registry.register_scenario(
        ScenarioRegistration(
            plugin=ScenarioPlugin(
                name="sql_harness",
                api_version=SCENARIO_PLUGIN_API_VERSION,
                factory=build_sql_harness_components,
            ),
            source="builtin",
        )
    )
    runtime = build_runtime(args.config, run_id=args.run_id, registry=registry)
    result = runtime.engine.run()
    print(json.dumps({"run_id": result.run_id, "selected_candidate_id": result.selected_candidate_id, "development_score": result.development_score, "iterations": result.iterations}, indent=2))


if __name__ == "__main__":
    main()
