"""Fixed ARE environment adapter for the shared OpenCode harness.

Run with the pinned Meta-ARE venv. The adapter owns events and evaluation;
AutoSaddler may change only the separate candidate harness directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import secrets
import threading
import time

from .accounting import UsageLedger
from .opencode_runtime import OpenCodeSession
from .gaia2_mcp_probe import REMOVED_AUI_TOOLS


class AREToolBridge:
    def __init__(self, scenario, env):
        from are.simulation.apps.agent_user_interface import AgentUserInterface
        from are.simulation.apps.mcp.server.are_simulation_mcp_server import ARESimulationToolManager
        from mcp.server.fastmcp import FastMCP
        from mcp.server.transport_security import TransportSecuritySettings
        self.env = env
        self.delivered = []
        self.manager = ARESimulationToolManager(apps=scenario.apps or [])
        allowed = {tool.name for tool in scenario.get_tools()} - REMOVED_AUI_TOOLS
        self.manager._tools = {k: v for k, v in self.manager._tools.items() if k in allowed}
        for app in scenario.apps or []:
            if isinstance(app, AgentUserInterface):
                app.wait_for_user_response = False
        for tool in self.manager._tools.values():
            tool.parameters = tool.fn_metadata.arg_model.model_json_schema()
            original = tool.fn
            async def wrapped(_original=original, **kwargs):
                result = await _original(**kwargs)
                pending = self.drain()
                if pending:
                    return {"tool_result": result, "environment_messages": pending}
                return result
            tool.fn = wrapped
        self.token = secrets.token_urlsafe(32)
        self.server = FastMCP("gaia2-environment", log_level="ERROR", transport_security=TransportSecuritySettings(
            allowed_hosts=["localhost:*", "127.0.0.1:*", "host.docker.internal:*"], allowed_origins=[]))
        self.server._tool_manager = self.manager

    def drain(self):
        timestamp = datetime.fromtimestamp(self.env.time_manager.time(), tz=timezone.utc)
        messages = self.env.notification_system.message_queue.get_by_timestamp(timestamp)
        values = [{"type": m.message_type.value, "text": m.message, "timestamp": m.timestamp.isoformat(),
                   "attachment_count": len(m.attachments)} for m in messages]
        # Attachments require a separate content mapping; fail rather than drop them.
        if any(m.attachments for m in messages):
            raise RuntimeError("GAIA2 attachment delivery is not implemented; do not score this task")
        self.delivered.extend(values)
        return values

    def __enter__(self):
        import uvicorn
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse
        token = self.token
        class Auth(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if request.headers.get("authorization") != "Bearer " + token:
                    return JSONResponse({"error": "unauthorized"}, status_code=401)
                return await call_next(request)
        self.socket = socket.socket()
        self.socket.bind(("0.0.0.0", 0))
        self.port = self.socket.getsockname()[1]
        app = self.server.streamable_http_app()
        app.add_middleware(Auth)
        self.http = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.http.run, kwargs={"sockets": [self.socket]}, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.http.started:
            if not self.thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("ARE MCP HTTP server failed to start")
            time.sleep(0.05)
        return self

    def mcp_config(self):
        return [{"type": "http", "name": "gaia2", "url": f"http://host.docker.internal:{self.port}/mcp", "headers": [{"name": "Authorization", "value": "Bearer " + self.token}]}]

    def __exit__(self, *args):
        self.http.should_exit = True
        self.thread.join(timeout=10)
        self.socket.close()


class OpenCodeAREAgent:
    def __init__(self, env, workspace, artifacts, model, ledger, run_id, role, task_id, timeout=1800, session_factory=OpenCodeSession):
        self.env, self.workspace, self.artifacts = env, workspace, artifacts
        self.model, self.ledger, self.run_id, self.role, self.task_id = model, ledger, run_id, role, task_id
        self.timeout, self.session_factory = timeout, session_factory

    def run_scenario(self, scenario, notification_system=None, initial_agent_logs=None):
        from are.simulation.agents.agent_execution_result import AgentExecutionResult
        from are.simulation.apps.agent_user_interface import AgentUserInterface
        from are.simulation.types import EnvironmentState
        if initial_agent_logs:
            raise ValueError("Resume from ARE agent logs is not implemented")
        aui = next(app for app in scenario.apps if isinstance(app, AgentUserInterface))
        deadline = time.monotonic() + self.timeout
        turns = 0
        with AREToolBridge(scenario, self.env) as bridge:
            overlay = {"permission": {"bash": "deny", "read": "deny", "edit": "deny", "write": "deny", "glob": "deny", "grep": "deny", "webfetch": "deny", "websearch": "deny", "external_directory": "deny", "gaia2_*": "allow"}}
            if self.model.startswith("codex-acp/"):
                from .acp_gateway import provider_config, context_headers
                overlay["provider"] = provider_config(self.model, headers=context_headers(self.run_id, self.role, "gaia2", self.task_id))
            with self.session_factory(self.workspace, self.artifacts, self.model, bridge.mcp_config(), overlay, container=True) as session:
                try:
                    while self.env.state not in {EnvironmentState.STOPPED, EnvironmentState.FAILED}:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("GAIA2 agent exceeded timeout")
                        messages = bridge.drain()
                        if any(m["type"] == "ENVIRONMENT_STOP" for m in messages):
                            break
                        if not messages:
                            # Explicit SystemApp.wait_for_notification tool calls use
                            # ARE's own time-jump mechanism. Idle polling must not call
                            # it without a configured timeout object.
                            time.sleep(0.05)
                            continue
                        before = len(aui.get_all_messages_from_agent())
                        prompt = "Environment messages:\n" + json.dumps(messages, ensure_ascii=False)
                        if turns == 0:
                            prompt = f"Simulated current time: {scenario.start_time}. Use only the GAIA2 tools. Send answers through AgentUserInterface.\n" + (scenario.additional_system_prompt or "") + "\n" + prompt
                        result = session.prompt(prompt, timeout=min(900, max(1, deadline - time.monotonic())))
                        turns += 1
                        # A natural-language final answer is an AUI action, never a score.
                        if result["text"].strip() and len(aui.get_all_messages_from_agent()) == before:
                            aui.send_message_to_user(result["text"])
                        # The official ARE main agent caps conversation turns at
                        # scenario.nb_turns. Final-turn validation happens after
                        # this return, in ScenarioRunner, not via ENVIRONMENT_STOP.
                        max_turns = getattr(scenario, "nb_turns", None)
                        if max_turns is not None and len(aui.get_all_messages_from_agent()) >= max_turns:
                            break
                finally:
                    session.export_usage(self.ledger, self.run_id, self.role, benchmark="gaia2", task_id=self.task_id)
                    self.artifacts.mkdir(parents=True, exist_ok=True)
                    (self.artifacts / "delivered_events.json").write_text(json.dumps(bridge.delivered, indent=2), encoding="utf-8")
        return AgentExecutionResult(output=aui.get_last_message_from_agent(), metadata={"turns": turns})

    def stop(self):
        self.env.stop()


class OpenCodeAREAgentBuilder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def list_agents(self):
        return ["default"]
    def build(self, agent_config, env=None, mock_responses=None):
        return OpenCodeAREAgent(env=env, **self.kwargs)


def run(config):
    from are.simulation.data_handler.importer import JsonScenarioImporter
    from are.simulation.scenario_runner import ScenarioRunner
    from are.simulation.scenarios.config import ScenarioRunnerConfig
    from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
    from are.simulation.scenarios.scenario_imported_from_json.utils import preprocess_scenario_from_config
    root = Path(__file__).resolve().parents[2]
    matches = list((root / "third_party/experiment_data/gaia2").rglob(f"*{config['task_id']}.json"))
    if len(matches) != 1:
        raise ValueError("GAIA2 task did not resolve uniquely")
    scenario, _, _ = JsonScenarioImporter().import_from_json_to_benchmark(matches[0].read_text(encoding="utf-8"), load_completed_events=False)
    artifact = Path(config["artifacts"])
    artifact.mkdir(parents=True, exist_ok=True)
    ledger = UsageLedger(Path(config["ledger"]))
    # All LiteLLM calls in this worker are evaluator/judge calls; the target
    # model runs in the isolated OpenCode process. Preserve request usage here.
    from are.simulation.agents.llm.litellm import litellm_engine
    import uuid
    completion = litellm_engine.completion
    def metered_completion(*args, **kwargs):
        request_id = "judge:" + uuid.uuid4().hex
        usage = {"input_tokens": None, "output_tokens": None, "total_tokens": None, "usage_complete": False}
        try:
            if config["model"].startswith("codex-acp/"):
                import os
                from .acp_gateway import context_headers
                kwargs["extra_headers"] = context_headers(config["run_id"], "judge", "gaia2", config["task_id"])
                kwargs["api_key"] = os.environ["ACP_GATEWAY_TOKEN"]
            response = completion(*args, **kwargs)
            raw = getattr(response, "usage", None)
            if raw is not None:
                usage = {"input_tokens": raw.prompt_tokens, "output_tokens": raw.completion_tokens,
                         "total_tokens": raw.total_tokens, "usage_complete": True}
            return response
        finally:
            ledger.record(request_id, config["run_id"], "judge", usage, benchmark="gaia2", task_id=config["task_id"], model=kwargs.get("model"))
    litellm_engine.completion = metered_completion
    judge_config = dict(config["judge"])
    if config["model"].startswith("codex-acp/"):
        import os
        judge_config["endpoint"] = os.environ["ACP_GATEWAY_LOCAL_URL"]
    runner_config = ScenarioRunnerConfig(agent="default", model=config["model"], model_provider="local", max_turns=None, export=True, output_dir=str(artifact),
        judge_engine_config=LLMEngineConfig(**judge_config))
    # Use the official preprocessing path (turn triggers, judge and assets).
    preprocess_scenario_from_config(scenario, runner_config)
    builder = OpenCodeAREAgentBuilder(workspace=Path(config["candidate"]), artifacts=artifact, model=config["model"], ledger=ledger, run_id=config["run_id"], role=config["role"], task_id=config["task_id"])
    result = ScenarioRunner(agent_builder=builder).run(runner_config, scenario)
    ledger.close()
    if result.exception is not None or result.success is None:
        raise RuntimeError(f"GAIA2 infrastructure/judge failure: {result.exception}")
    return {"score": float(result.success), "valid_evaluation": True, "validation_rationale": result.rationale, "trace_path": result.export_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.config.read_text(encoding="utf-8")))
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
