"""Task-scoped MCP tools for temporary Codex ACP integration checks.

Run with the ARE venv. Docker tools address one explicitly selected disposable
container; no workspace, Docker socket, or host credentials are mounted in it.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def docker_server(container, audit, cwd=None, max_calls=20):
    from mcp.server.fastmcp import FastMCP
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", container):
        raise ValueError("Invalid fixed container identifier")
    server = FastMCP("acp-disposable-task")
    calls = 0

    @server.tool()
    def execute(command: str) -> dict:
        """Run a shell command inside the disposable Linux task container.

        Use this tool for all task file inspection, editing, and testing.
        Commands have a 45-second limit and at most 20 calls per task.
        """
        nonlocal calls
        calls += 1
        if calls > max_calls:
            raise RuntimeError("Task tool-call limit reached; stop the smoke test")
        # The identifier is fixed by the trusted test runner, never by the model.
        argv = ["docker", "exec"]
        if cwd:
            argv += ["-w", cwd]
        argv += [container, "/bin/sh", "-c", command]
        try:
            completed = subprocess.run(argv, capture_output=True, encoding="utf-8", errors="replace", timeout=45)
            result = {"exit_code": completed.returncode, "stdout": completed.stdout[-18000:], "stderr": completed.stderr[-4000:]}
        except subprocess.TimeoutExpired:
            result = {"exit_code": None, "error": "Command timed out; do not assume it completed"}
        append(audit, {"call": calls, "command": command, "result": result})
        return result
    return server


def gaia_server(audit):
    from . import gaia2_mcp_probe as probe
    root = Path(__file__).resolve().parents[2]
    task_id = "scenario_universe_29_096dyu"
    manifest = json.loads(Path(__file__).with_name("manifests").joinpath("gaia2.json").read_text())
    if task_id not in manifest["train"]:
        raise ValueError("ACP probe must use a training scenario")
    matches = list((root / "third_party/experiment_data/gaia2").rglob("*" + task_id + ".json"))
    if len(matches) != 1:
        raise ValueError("Training scenario did not resolve uniquely")
    probe.SCENARIO = matches[0]
    probe.OUT = audit.parent / "are_environment"
    server, scenario, env, details = probe.build_probe_server()
    keep = {"Calendar__search_events", "Calendar__add_calendar_event", "AgentUserInterface__send_message_to_user"}
    manager = server._tool_manager
    manager._tools = {name: tool for name, tool in manager._tools.items() if name in keep}
    if set(manager._tools) != keep:
        raise ValueError("Unexpected calendar tool names")
    for name, tool in manager._tools.items():
        original = tool.fn
        async def wrapped(_original=original, _name=name, **kwargs):
            result = await _original(**kwargs)
            append(audit, {"tool": _name, "arguments": kwargs, "result": result})
            return result
        tool.fn = wrapped
    append(audit, {"scenario_id": scenario.scenario_id, "split": "train", "official_score": False, "exposed_tools": sorted(keep)})
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["gaia", "docker"], required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--container")
    parser.add_argument("--cwd")
    args = parser.parse_args()
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "gaia":
        server = gaia_server(args.audit)
    else:
        if not args.container:
            parser.error("--container is required")
        server = docker_server(args.container, args.audit, args.cwd)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
