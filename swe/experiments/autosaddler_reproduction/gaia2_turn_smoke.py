"""Model-free regression: a completed final ARE turn must return for judging."""
from pathlib import Path
import json
import tempfile


def main():
    from are.simulation.apps.agent_user_interface import AgentUserInterface
    from are.simulation.apps.calendar import CalendarApp
    from are.simulation.environment import Environment, EnvironmentConfig
    from are.simulation.scenarios.scenario import Scenario
    from are.simulation.types import EnvironmentState
    from .gaia2_adapter import OpenCodeAREAgent
    class Fixture(Scenario):
        nb_turns = 1
        def init_and_populate_apps(self, **kwargs):
            self.apps = [AgentUserInterface(), CalendarApp()]
        def build_events_flow(self):
            self.events = []
    fixture = Fixture(scenario_id="synthetic_final_turn", start_time=1728975600.0, duration=120)
    fixture.initialize()
    fixture.nb_turns = 1
    env = Environment(EnvironmentConfig(start_time=fixture.start_time, verbose=False))
    env.run(fixture, wait_for_end=False)
    aui = fixture.apps[0]
    aui.send_message_to_agent("Send one completion message.")
    count = []
    class Session:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def prompt(self, *args, **kwargs):
            count.append(1)
            aui.send_message_to_user("Done.")
            return {"text": "Done."}
        def export_usage(self, *args, **kwargs):
            return 0  # This fixture never invokes an LLM.
    root = Path(__file__).resolve().parents[2]
    artifacts = root / "results/autosaddler_reproduction/infrastructure/gaia2_final_turn"
    artifacts.mkdir(parents=True, exist_ok=True)
    try:
        agent = OpenCodeAREAgent(env, root, artifacts, "fixture/model", None, "turn-regression", "target_search", fixture.scenario_id, timeout=30, session_factory=Session)
        result = agent.run_scenario(fixture)
        assert len(count) == 1 and result.metadata["turns"] == 1
        assert env.state == EnvironmentState.RUNNING, "Agent must return for official judging, not force-stop the environment"
        assert len(aui.get_all_messages_from_agent()) == 1, "Final text duplicated an explicit AUI message"
        report = {"passed": True, "model_requests": 0, "returns_after_declared_final_turn": True, "no_duplicate_aui": True, "official_score": False}
        (artifacts / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
    finally:
        env.stop()


if __name__ == "__main__":
    main()
