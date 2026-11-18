"""Exercise the real ACP path using the smallest available local Codex model.

This is an infrastructure test with a different agent, never an OpenCode score.
"""
import json
import os
from pathlib import Path
import shutil

from .acp_client import ACPClient

ROOT = Path(__file__).resolve().parents[2]


def main():
    work = ROOT / ".test_tmp/acp_model_smoke"
    work.mkdir(parents=True, exist_ok=True)
    (work / "fixture.txt").write_text("left=17\nright=25\n", encoding="utf-8")
    artifacts = ROOT / "results/autosaddler_reproduction/acp_model_smoke"
    node = shutil.which("node")
    codex = shutil.which("codex")
    if not node or not codex:
        raise RuntimeError("Node and authenticated Codex CLI are required")
    adapter = ROOT / "third_party/codex-acp/node_modules/@agentclientprotocol/codex-acp/dist/index.js"
    env = {**os.environ, "CODEX_PATH": codex, "INITIAL_AGENT_MODE": "read-only",
           "CODEX_CONFIG": json.dumps({"model": "gpt-6-luna", "model_reasoning_effort": "low", "approval_policy": "never", "sandbox_mode": "read-only"})}
    with ACPClient([node, str(adapter)], work, artifacts, env) as client:
        init = client.initialize()
        session = client.new_session()
        session_id = session["sessionId"]
        (artifacts / "capabilities.json").write_text(json.dumps({"initialize": init, "session": session}, indent=2), encoding="utf-8")
        configs = session.get("configOptions", [])
        model_config = next((c for c in configs if c.get("category") == "model" or c.get("id") == "model"), None)
        if model_config:
            client.request("session/set_config_option", {"sessionId": session_id, "configId": model_config["id"], "value": "gpt-6-luna"})
        elif session.get("models"):
            client.request("session/set_model", {"sessionId": session_id, "modelId": "gpt-6-luna"})
        fixture = (work / "fixture.txt").read_text(encoding="utf-8")
        answer = client.prompt(session_id, "Infrastructure smoke test. Fixture content supplied by the ACP client:\n" + fixture + "\nAdd the left and right integers. Reply with exactly one JSON object containing the answer key. Do not use tools, edit files or access the network.", timeout=180)
        response = answer["text"].strip()
        parsed = json.loads(response.removeprefix("```json").removesuffix("```").strip())
        if parsed != {"answer": 42}:
            raise AssertionError(response)
        result = {"passed": True, "agent": "codex-acp", "model": "gpt-6-luna", "purpose": "ACP infrastructure smoke only", "is_opencode_benchmark_result": False, "response": parsed, "acp_stop_reason": answer["result"].get("stopReason")}
        (artifacts / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
