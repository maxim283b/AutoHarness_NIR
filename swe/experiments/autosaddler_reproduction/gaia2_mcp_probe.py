"""Credential-free GAIA2/OpenCode tool transport probe, NOT a benchmark runner.

Use the Meta-ARE venv Python. The worker loads a real train scenario in memory,
without scheduling its events, invoking an LLM, or computing a benchmark score.
The hardened probe server exposes only agent tools, no raw app-state resources.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "third_party/Meta-ARE/datasets_local/gaia2_smoke/adaptability/validation/0042_scenario_universe_29_096dyu.json"
OUT = ROOT / "results/autosaddler_reproduction/gaia2_compatibility"
REMOVED_AUI_TOOLS = {
    "AgentUserInterface__get_last_message_from_user",
    "AgentUserInterface__get_last_message_from_agent",
    "AgentUserInterface__get_last_unread_messages",
    "AgentUserInterface__get_all_messages",
}


def build_probe_server():
    from are.simulation.data_handler.importer import JsonScenarioImporter
    from are.simulation.environment import Environment, EnvironmentConfig
    from are.simulation.apps.agent_user_interface import AgentUserInterface
    from are.simulation.apps.mcp.server.are_simulation_mcp_server import ARESimulationToolManager
    from mcp.server.fastmcp import FastMCP
    from jsonschema import Draft202012Validator

    # Direct importer avoids unrelated Hugging Face registry discovery.
    scenario, _, _ = JsonScenarioImporter().import_from_json(
        SCENARIO.read_text(encoding="utf-8"), load_completed_events=False
    )
    (OUT / "sandbox").mkdir(parents=True, exist_ok=True)
    scenario.initialize(sandbox_dir=str(OUT / "sandbox"))
    env = Environment(EnvironmentConfig(start_time=scenario.start_time, verbose=False))
    env.register_apps(scenario.apps or [])
    for app in scenario.apps or []:
        if isinstance(app, AgentUserInterface):
            app.wait_for_user_response = False
    manager = ARESimulationToolManager(apps=scenario.apps or [])
    allowed = {tool.name for tool in scenario.get_tools()} - REMOVED_AUI_TOOLS
    manager._tools = {name: tool for name, tool in manager._tools.items() if name in allowed}
    invalid_stock = []
    for name, tool in manager._tools.items():
        try:
            Draft202012Validator.check_schema(tool.parameters)
        except Exception:
            invalid_stock.append(name)
        # Use the same Pydantic argument contract as the upstream executor.
        tool.parameters = tool.fn_metadata.arg_model.model_json_schema()
        Draft202012Validator.check_schema(tool.parameters)
    server = FastMCP("gaia2-compatibility-probe")
    server._tool_manager = manager
    details = {
        "scenario_id": scenario.scenario_id,
        "split": "paper train universe 29",
        "apps": len(scenario.apps or []),
        "scheduled_event_definitions": len(scenario.events or []),
        "events_executed": False,
        "agent_tools": len(allowed),
        "mcp_tools": len(manager._tools),
        "tool_set_matches": set(manager._tools) == allowed,
        "stock_schema_invalid_tools": invalid_stock,
        "normalized_schemas_valid": True,
        "raw_state_resources_exposed": False,
        "warning": "Tool transport probe only; no event loop, official judge or OpenCode/model task execution",
    }
    return server, scenario, env, details


async def client_probe():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).resolve()), "--serve"],
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
    )
    async with stdio_client(params) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            resources = (await session.list_resources()).resources
            read_name = next(t.name for t in tools if t.name.endswith("__search_events"))
            write_name = next(t.name for t in tools if t.name.endswith("__add_calendar_event"))
            title = "AUTOSADDLER_COMPATIBILITY_PROBE_20261102"
            before = await session.call_tool(read_name, {"query": title})
            write = await session.call_tool(write_name, {
                "title": title,
                "start_datetime": "2024-10-16 10:00:00",
                "end_datetime": "2024-10-16 10:30:00",
            })
            after = await session.call_tool(read_name, {"query": title})
            before_text = json.dumps(before.model_dump(mode="json"))
            after_text = json.dumps(after.model_dump(mode="json"))
            result = {
                "initialize": True,
                "listed_tools": len(tools),
                "resources_empty": not resources,
                "read_success": not before.isError,
                "write_success": not write.isError,
                "read_after_write": not after.isError and title in after_text and title not in before_text,
                "inference_requests": 0,
                "official_benchmark_evaluated": False,
            }
            assert all(result[k] for k in ("resources_empty", "read_success", "write_success", "read_after_write")), result
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--client", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
    if args.serve or args.inspect:
        server, scenario, env, details = build_probe_server()
        if args.serve:
            server.run(transport="stdio")
        else:
            print(json.dumps(details, indent=2))
    elif args.client:
        print(json.dumps(asyncio.run(asyncio.wait_for(client_probe(), timeout=90)), indent=2))
    else:
        # Keep native imports and filesystem setup bounded; preserve diagnostics.
        for mode, filename in (("--inspect", "inspection.json"), ("--client", "mcp_roundtrip.json")):
            try:
                completed = subprocess.run(
                    [sys.executable, "-u", str(Path(__file__).resolve()), mode],
                    capture_output=True, text=True, encoding="utf-8", timeout=120,
                )
            except subprocess.TimeoutExpired:
                print("Probe subprocess timed out; Windows stdio may require execution outside the tool sandbox", file=sys.stderr)
                return 1
            (OUT / (filename + ".stderr.log")).write_text(completed.stderr, encoding="utf-8")
            if completed.returncode:
                print(completed.stderr[-4000:], file=sys.stderr)
                return completed.returncode
            data = json.loads(completed.stdout)
            (OUT / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(data, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
