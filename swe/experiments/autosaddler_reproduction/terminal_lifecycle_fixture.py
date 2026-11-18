"""Exercise Harbor's actual timeout -> cleanup -> verifier path without a model."""
import json
import os
from pathlib import Path
import subprocess
import uuid

from harbor.models.job.config import JobConfig

from .terminal_adapter import SharedOpenCode, ROOT, trial_outcome


class FixtureAgent(SharedOpenCode):
    async def install(self, environment):
        # No model binary or credentials are installed in this fixture.
        await environment.exec(command="mkdir -p /logs/agent ~/.config/opencode")

    async def exec_as_agent(self, environment, command, env=None, **kwargs):
        if "opencode --model=" in command:
            command = "setsid bash -c 'echo $$ > /tmp/marked-child.pid; sleep 300 & wait' & wait"
        elif not command.startswith("mkdir -p "):
            raise RuntimeError("Unexpected command in offline lifecycle fixture")
        return await super().exec_as_agent(environment, command, env=env, **kwargs)


def main():
    task = ROOT / ".test_tmp/terminal-lifecycle-fixture"
    (task / "environment").mkdir(parents=True, exist_ok=True)
    (task / "tests").mkdir(exist_ok=True)
    (task / "instruction.md").write_text("Offline process lifecycle test.", encoding="utf-8")
    (task / "task.toml").write_text('''version = "1.0"
[agent]
timeout_sec = 10.0
[verifier]
timeout_sec = 30.0
[environment]
docker_image = "alexgshaw/compile-compcert:20251031"
build_timeout_sec = 120.0
cpus = 1
memory_mb = 512
''', encoding="utf-8")
    (task / "environment/Dockerfile").write_bytes(b"FROM alexgshaw/compile-compcert:20251031\n")
    (task / "tests/test.sh").write_bytes(b'''#!/bin/bash
mkdir -p /logs/verifier
if [ -f /logs/agent/autoharness_cleanup.json ] && [ -f /tmp/marked-child.pid ] && ! kill -0 $(cat /tmp/marked-child.pid) 2>/dev/null; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
''')
    artifacts = ROOT / "results/autosaddler_reproduction/lifecycle_fixture" / uuid.uuid4().hex[:10]
    artifacts.mkdir(parents=True)
    config = JobConfig.model_validate({"job_name":"trial", "jobs_dir":str(artifacts), "n_attempts":1,
        "n_concurrent_trials":1, "skip_credential_checks":True, "environment":{"type":"docker","delete":True},
        "agents":[{"import_path":"experiments.autosaddler_reproduction.terminal_lifecycle_fixture:FixtureAgent",
            "model_name":"fixture/model", "skip_model_parsing":True, "kwargs":{"candidate_dir":str(ROOT / "experiments/autosaddler_reproduction/harness_seed"),"version":"1.19.5"}}],
        "tasks":[{"path":str(task)}]})
    config_path = artifacts / "job.json"
    config_path.write_text(config.model_dump_json(indent=2),encoding="utf-8")
    env = os.environ.copy()
    for key in ("GO_GATEWAY_URL","OPENCODE_GO_API_KEY","OPENCODE_API_KEY","ACP_GATEWAY_TOKEN"):
        env.pop(key, None)
    env.update(PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
    done = subprocess.run([str(ROOT / ".venv-experiment/Scripts/harbor.exe"),"run","--config",str(config_path)],
        capture_output=True,text=True,encoding="utf-8",errors="replace",env=env,timeout=240)
    (artifacts / "stdout.log").write_text(done.stdout,encoding="utf-8")
    (artifacts / "stderr.log").write_text(done.stderr,encoding="utf-8")
    results = list((artifacts / "trial").glob("*/result.json"))
    if done.returncode or len(results)!=1:
        raise RuntimeError("Offline Harbor lifecycle fixture failed: " + str(artifacts))
    trial = json.loads(results[0].read_text())
    cleanup = results[0].parent / "agent/autoharness_cleanup.json"
    outcome = trial_outcome(trial, cleanup_ok=cleanup.exists() and json.loads(cleanup.read_text()).get("cleanup_ok") is True)
    if outcome["score"]!=1 or not outcome["agent_timed_out"]:
        raise RuntimeError("Expected timed-out agent with passing cleanup verifier: " + str(artifacts))
    proof = {"passed":True,"real_docker":True,"real_harbor":True,"agent_timed_out":True,
        "verifier_reward":outcome["score"],"cleanup_before_verifier":True,"model_calls":0,"artifacts":str(artifacts)}
    (ROOT / "results/autosaddler_reproduction/terminal_cleanup_proof.json").write_text(json.dumps(proof,indent=2),encoding="utf-8")
    print(json.dumps(proof,indent=2))


if __name__ == "__main__":
    main()
