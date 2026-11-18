"""Validate a real GAIA2 train scenario and local assets using native oracle replay."""
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    os.environ["DEMO_FS_PATH"] = str(root / "third_party/meta_are_data/gaia2_filesystem/132e26376f5e963bb59f64bcccdd02188cb08dee/demo_filesystem")
    from are.simulation.data_handler.importer import JsonScenarioImporter
    from are.simulation.scenario_runner import ScenarioRunner
    from are.simulation.scenarios.config import ScenarioRunnerConfig
    from are.simulation.scenarios.scenario_imported_from_json.utils import preprocess_scenario_from_config
    manifest = json.loads((root / "third_party/AutoSaddler/configs/datasets/GAIA2/train.json").read_text())
    task_id = manifest["execution"][0]
    scenario_path = next((root / "third_party/experiment_data/gaia2").rglob(f"*{task_id}.json"))
    scenario, _, _ = JsonScenarioImporter().import_from_json(scenario_path.read_text(encoding="utf-8"), load_completed_events=False)
    output = root / "results/autosaddler_reproduction/infrastructure/gaia2_oracle"
    output.mkdir(parents=True, exist_ok=True)
    config = ScenarioRunnerConfig(agent=None, oracle=True, export=True, output_dir=str(output))
    preprocess_scenario_from_config(scenario, config)
    result = ScenarioRunner().run(config, scenario)
    if result.exception is not None:
        raise RuntimeError(str(result.exception))
    report = {"task_id":task_id, "native_oracle_replay":True, "model_requests":0,
              "official_agent_score":False, "result":result.success, "trace_path":result.export_path,
              "note":"Checks native scenario/actions/assets. LLM judge binding is tested after model configuration."}
    (output / "result.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
