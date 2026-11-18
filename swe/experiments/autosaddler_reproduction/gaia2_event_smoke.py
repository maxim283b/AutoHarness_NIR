"""Native ARE lifecycle + real containerized OpenCode + local scripted model."""
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

from .accounting import UsageLedger
from .gaia2_adapter import OpenCodeAREAgent
from .mock_model import MockModel
from .opencode_runtime import OpenCodeSession


def main():
    from are.simulation.apps.agent_user_interface import AgentUserInterface
    from are.simulation.apps.calendar import CalendarApp
    from are.simulation.environment import Environment, EnvironmentConfig
    from are.simulation.scenarios.scenario import Scenario
    from are.simulation.notification_system import Message, MessageType
    root = Path(__file__).resolve().parents[2]
    artifacts = root / "results/autosaddler_reproduction/infrastructure/gaia2_event_smoke"
    artifacts.mkdir(parents=True, exist_ok=True)
    workspace = root / ".test_tmp/gaia2-event-smoke/workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    for name in ("opencode.json", "instructions.md"):
        shutil.copy2(Path(__file__).with_name("harness_seed") / name, workspace / name)
    class Fixture(Scenario):
        def init_and_populate_apps(self, **kwargs):
            self.apps = [AgentUserInterface(), CalendarApp()]
        def build_events_flow(self):
            self.events = []
    scenario = Fixture(scenario_id="synthetic_opencode_are_lifecycle", start_time=1728975600.0, duration=120)
    scenario.initialize()
    env = Environment(EnvironmentConfig(start_time=scenario.start_time, verbose=False))
    env.run(scenario, wait_for_end=False)
    aui, calendar = scenario.apps
    aui.send_message_to_agent("Create a calendar entry titled infrastructure-test, verify it, then report completion.")
    phase = {"n": 0}
    def responder(payload, request_number):
        tools = payload.get("tools") or []
        # OpenCode may make auxiliary title/summary requests without task tools.
        if not tools:
            return {"content":"Infrastructure test"}
        phase["n"] += 1
        index = phase["n"]
        if index == 1:
            env.notification_system.message_queue.put(Message(message_type=MessageType.ENVIRONMENT_NOTIFICATION, message="MID_TURN_NOTIFICATION", timestamp=datetime.fromtimestamp(env.time_manager.time(), timezone.utc)))
            suffix = "__add_calendar_event"
            args = {"title":"infrastructure-test", "start_datetime":"2024-10-16 10:00:00", "end_datetime":"2024-10-16 10:30:00"}
        elif index == 2:
            suffix, args = "__search_events", {"query":"infrastructure-test"}
        elif index == 3:
            suffix, args = "AgentUserInterface__send_message_to_user", {"content":"Infrastructure check completed."}
        else:
            env.stop()
            return {"content":"Infrastructure check completed."}
        name = next(t["function"]["name"] for t in tools if t["function"]["name"].endswith(suffix))
        return {"tool_calls":[{"index":0,"id":f"call-{index}","type":"function","function":{"name":name,"arguments":json.dumps(args)}}]}
    ledger = UsageLedger(artifacts / "usage.sqlite")
    try:
        with MockModel(responder=responder) as fixture:
            providers = fixture.provider()
            providers["mock"]["options"]["baseURL"] = fixture.base_url.replace("127.0.0.1", "host.docker.internal")
            def session_factory(work, out, model, mcp, overlay, **kwargs):
                return OpenCodeSession(work, out, "mock/fixture", mcp, {**overlay, "provider":providers}, **kwargs)
            agent = OpenCodeAREAgent(env, workspace, artifacts, "mock/fixture", ledger, "gaia2-event-smoke", "target_search", scenario.scenario_id, timeout=90, session_factory=session_factory)
            agent.run_scenario(scenario)
        delivered = json.loads((artifacts / "delivered_events.json").read_text())
        assert calendar.search_events("infrastructure-test"), "MCP calendar mutation did not occur"
        assert any(m["text"] == "MID_TURN_NOTIFICATION" for m in delivered), "Dynamic notification was dropped"
        assert aui.get_all_messages_from_agent(), "No answer went through AUI"
        from are.simulation.types import EnvironmentState
        assert env.state == EnvironmentState.STOPPED
        result = {"passed":True, "synthetic_scenario":True, "official_gaia2_score":False, "real_opencode_in_container":True,
            "mcp_write_read_and_aui":True, "mid_turn_notification_delivered":True, "environment_stopped":True,
            "model_requests_paid":0, "usage":ledger.totals("gaia2-event-smoke")}
        (artifacts / "result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps(result,indent=2))
    finally:
        env.stop()
        ledger.close()


if __name__ == "__main__":
    main()
