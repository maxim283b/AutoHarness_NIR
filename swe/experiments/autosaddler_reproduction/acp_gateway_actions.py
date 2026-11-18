"""Mirror OpenCode tool schemas as MCP action-selection tools.

These tools ONLY record a requested action. They never execute commands, read
task files, contact a service, or modify the benchmark environment. OpenCode is
the sole executor after it receives the recorded decision through its provider.
"""
import argparse
import asyncio
import json
from pathlib import Path


async def serve(tools_path, actions_path):
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, ToolAnnotations, TextContent
    definitions = json.loads(tools_path.read_text(encoding="utf-8"))
    available = {item["function"]["name"]: item["function"] for item in definitions}
    server = Server("opencode-action-selection")
    lock = asyncio.Lock()
    selected = False

    @server.list_tools()
    async def list_tools():
        return [Tool(name=name, description="Select this action for the external OpenCode executor. This tool records intent only; stop after one selection.\n" + definition.get("description", ""),
            inputSchema=definition.get("parameters", {"type": "object"}), annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False)) for name, definition in available.items()]

    @server.call_tool()
    async def call_tool(name, arguments):
        nonlocal selected
        async with lock:
            if name not in available:
                raise ValueError("Unknown external tool")
            if selected:
                return [TextContent(type="text", text="An action has already been selected. End this turn now; OpenCode will supply the real result in the next conversation.")]
            with actions_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"name": name, "arguments": arguments or {}}, ensure_ascii=False) + "\n")
            selected = True
            return [TextContent(type="text", text="Action selection recorded for OpenCode. No task action has run yet. End this turn now. The actual tool result will appear in the next supplied conversation.")]

    async with stdio_server() as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(serve(args.tools, args.actions))
