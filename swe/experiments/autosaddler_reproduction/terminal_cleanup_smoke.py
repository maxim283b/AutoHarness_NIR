"""Real Docker cancellation/descendant cleanup proof, with no model or API calls."""
import asyncio
import json
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .terminal_adapter import SharedOpenCode, OpenCode, ROOT


async def exercise(name):
    class Environment:
        async def exec(self, command, env=None, **kwargs):
            cmd = ["docker", "exec"]
            for key, value in (env or {}).items():
                cmd += ["-e", key+"="+value]
            process = await asyncio.create_subprocess_exec(*cmd, name, "bash", "-c", command,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            # On cancellation leave the container process for the tested cleanup
            # method, matching the actual Harbor/Docker behavior that failed.
            stdout, stderr = await process.communicate()
            return SimpleNamespace(return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode())
    environment = Environment()
    async def upstream_exec(self, environment, command, env=None, **kwargs):
        return await environment.exec(command=command, env=env, **kwargs)
    async def fake_agent(self, instruction, environment, context):
        await self.exec_as_agent(environment, command="nohup sleep 300 >/dev/null 2>&1 </dev/null & echo $! > /tmp/completed-service.pid")
        await self.exec_as_agent(environment, command="setsid bash -c 'echo $$ > /tmp/marked-child.pid; sleep 300 & wait' & wait")
    with patch.object(OpenCode, "exec_as_agent", upstream_exec), patch.object(OpenCode, "run", fake_agent):
        try:
            await asyncio.wait_for(object.__new__(SharedOpenCode).run("fixture", environment, None), timeout=4)
        except TimeoutError:
            pass
        else:
            raise AssertionError("Fixture was expected to time out")
    result = await environment.exec(command="test -f /tmp/marked-child.pid && test -f /logs/agent/autoharness_cleanup.json && ! kill -0 $(cat /tmp/marked-child.pid) 2>/dev/null && kill -0 $(cat /tmp/unmarked.pid) && kill -0 $(cat /tmp/completed-service.pid)")
    if result.return_code:
        raise RuntimeError("Detached child survived or unrelated sentinel was killed: " + result.stderr)
    return {"passed":True,"real_docker":True,"timed_out_child_terminated":True,"unmarked_process_preserved":True,"completed_service_preserved":True,"model_calls":0}


def main():
    name = "autoharness-cleanup-proof-" + uuid.uuid4().hex[:10]
    subprocess.run(["docker", "run", "--rm", "-d", "--name", name, "autoharness-opencode:1.19.5",
        "bash", "-c", "sleep 300 & echo $! > /tmp/unmarked.pid; wait"],check=True,capture_output=True)
    try:
        proof = asyncio.run(exercise(name))
        path = ROOT / "results/autosaddler_reproduction/terminal_cleanup_proof.json"
        path.write_text(json.dumps(proof,indent=2),encoding="utf-8")
        print(json.dumps(proof))
    finally:
        subprocess.run(["docker","rm","-f",name],capture_output=True,timeout=30)


if __name__ == "__main__":
    main()
