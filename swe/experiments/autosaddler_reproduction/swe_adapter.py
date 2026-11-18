"""OpenCode generation in a container and fresh official SWE-Bench Pro V1 grading."""
from __future__ import annotations

import ast
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import threading
import uuid
from unittest.mock import patch as mock_patch

from .accounting import normalize_opencode

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/SWE-bench_Pro-os"
_PINNED_GRADER_LOCK = threading.Lock()


def load_row(task_id):
    path = ROOT / "third_party/experiment_data/swe_bench_pro_v1.jsonl"
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row["instance_id"] == task_id:
            return row
    raise ValueError(f"Unknown SWE task {task_id}")


def grading_container_limits(row):
    limits=dict(network_mode='none',platform='linux/amd64',mem_limit='6g',pids_limit=512,remove=False)
    if row.get('repo')=='element-hq/element-web':
        # Jest 29 uses Node's availableParallelism: two allowed CPUs mean one
        # worker, instead of nine workers exhausting this Docker VM's RAM.
        limits['cpuset_cpus']='0-1'
    return limits


def verify_execution_limits(row, execution, *, legacy=False):
    expected=grading_container_limits(row)
    if legacy and 'cpuset_cpus' in expected:
        raise RuntimeError('Legacy control used a different CPU profile')
    fields={'cpuset_cpus':expected.get('cpuset_cpus',''),'memory_limit_bytes':6*1024**3,'pids_limit':512}
    if any(execution.get(k,v if legacy else None)!=v for k,v in fields.items()):
        raise RuntimeError('Official grader resource profile differs from the registered host limits')


def grade_patch(row, patch, artifacts, *, image=None):
    sys.path.insert(0, str(SOURCE))
    grader = importlib.import_module("swe_bench_pro_eval")
    def write_files_lf(directory, files):
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.replace("\r\n", "\n").encode("utf-8"))
    # Upstream text-mode writes introduce CRLF on Windows, breaking bash scripts.
    grader.write_files_local = write_files_lf
    artifacts = artifacts.resolve()
    # Upstream redo bypasses its top-level cache but reuses workspace/. A
    # previously written output.json could survive an interrupted entryscript.
    # Remove only this attempt's grading workspace and output before rerunning.
    task_artifacts = (artifacts / row["instance_id"]).resolve()
    if not task_artifacts.is_relative_to(artifacts):
        raise RuntimeError("SWE grading task path escapes its attempt directory")
    stale_workspace = task_artifacts / "workspace"
    stale_output = task_artifacts / "candidate_output.json"
    for owned in (stale_workspace, stale_output):
        if not owned.resolve().is_relative_to(artifacts):
            raise RuntimeError("SWE grading cache path escapes its attempt directory")
    if stale_workspace.exists():
        shutil.rmtree(stale_workspace)
    if stale_output.exists():
        stale_output.unlink()
    clients = []
    containers = []
    executions = []
    if image is not None:
        if not isinstance(image, dict) or not image.get("image_id") or not image.get("tag"):
            raise RuntimeError("Official SWE grading requires a frozen image ID and tag")
        expected_tag, image_id = image["tag"], image["image_id"]
        actual_tag = grader.get_dockerhub_image_uri(row["instance_id"], "jefzda", row.get("repo", ""))
        if actual_tag != expected_tag:
            raise RuntimeError("Frozen SWE image tag differs from the official evaluator's tag")
        original_from_env = grader.docker.from_env

        def pinned_from_env(*args, **kwargs):
            client = original_from_env(*args, **kwargs)
            clients.append(client)
            client.images.get(image_id)  # The pinned image must be present before grading.

            class PinnedImages:
                def pull(self, tag, *pull_args, **pull_kwargs):
                    if tag != expected_tag:
                        raise RuntimeError("Official evaluator requested another image tag")
                    return client.images.get(image_id)

                def get(self, tag):
                    if tag != expected_tag:
                        raise RuntimeError("Official evaluator requested another image tag")
                    return client.images.get(image_id)

            class PinnedContainers:
                def run(self, tag, **run_kwargs):
                    if tag != expected_tag:
                        raise RuntimeError("Official evaluator requested another image tag")
                    run_kwargs.update(grading_container_limits(row))
                    container = client.containers.run(image_id, **run_kwargs)
                    containers.append(container)
                    return container

            class PinnedClient:
                images = PinnedImages()
                containers = PinnedContainers()

            return PinnedClient()

    previous_cwd = Path.cwd()
    try:
        os.chdir(SOURCE)  # Upstream assembles Dockerfiles relative to its repo.
        # A prior attempt may have written candidate_output.json before its
        # result/provenance was committed. Always regrade the actual patch on
        # retry; runner-level validated result.json handles safe resume.
        if image is None:
            output = grader.eval_with_docker(patch, row, str(artifacts), "jefzda", str(SOURCE / "run_scripts"), prefix="candidate", redo=True)
        else:
            with _PINNED_GRADER_LOCK, mock_patch.object(grader.docker, "from_env", pinned_from_env):
                output = grader.eval_with_docker(patch, row, str(artifacts), "jefzda", str(SOURCE / "run_scripts"), prefix="candidate", redo=True)
    finally:
        os.chdir(previous_cwd)
        for container in containers:
            try:
                container.reload()
                state = container.attrs['State']
                executions.append({'image_id':container.attrs['Image'],'exit_code':state['ExitCode'],
                    'oom_killed':state['OOMKilled'],'running':state['Running'],
                    'started_at':state['StartedAt'],'finished_at':state['FinishedAt'],'network':'none',
                    'cpuset_cpus':container.attrs['HostConfig']['CpusetCpus'],
                    'memory_limit_bytes':container.attrs['HostConfig']['Memory'],
                    'pids_limit':container.attrs['HostConfig']['PidsLimit']})
            finally:
                container.remove(force=True)
        for client in clients:
            client.close()
    if image is not None:
        (artifacts/'execution.json').write_text(json.dumps({'executions':executions,'cleanup_verified':True},indent=2)+'\n')
        if len(executions) != 1 or any(e['exit_code'] != 0 or e['oom_killed'] or e['running'] for e in executions):
            raise RuntimeError('Official grader did not complete normally; technical NOSCORE')
        for execution in executions:verify_execution_limits(row,execution)
    return score_official_artifacts(row, output, artifacts)


def score_official_artifacts(row, output, artifacts):
    """Interpret retained diagnostics without rerunning the official grader."""
    task_artifacts = Path(artifacts) / row['instance_id']
    logs = ''
    for name in ('candidate_stdout.log','candidate_stderr.log'):
        path = task_artifacts/name
        if path.exists(): logs += path.read_text(encoding='utf-8',errors='replace')
    if (row.get('repo') in {'flipt-io/flipt','future-architect/vuls'}
        and re.search(r'Get "https://(?:proxy|sum)\.golang\.org/',logs)
        and any(marker in logs for marker in ('network is unreachable','dial tcp:','TLS handshake timeout'))):
        raise RuntimeError('Official Go verifier could not access a dependency; technical NOSCORE')
    # A task implementation can fail to compile before its tests run. The
    # official SWE predicate grades that candidate as false; this differs from
    # failure to build the OpenCode harness itself or to start the verifier.
    build_failure = any(marker in logs for marker in (
        '[build failed]','SyntaxError:','COMPILATION ERROR','Test suite failed to run','error TS'))
    # Some tasks add a repository-owned module. Its absence in the base tree
    # stops pytest collection, which is an official task failure. Do not
    # generalize this to missing third-party packages or a broken verifier.
    if row.get('repo') == 'ansible/ansible':
        build_failure = build_failure or bool(re.search(
            r"(?:ImportError: cannot import name '[\w]+' from |ModuleNotFoundError: No module named )"
            r"'ansible\.(?:modules|module_utils|plugins)\.[\w.]+['\"]", logs))
    # The pinned Go parser emits only passing top-level tests. A real
    # assertion failure can therefore leave its parsed list empty. Retain
    # the official predicate (false) when Go's own test runner reported a
    # failed test; dependency/download failures do not match this marker.
    if row.get('repo') == 'flipt-io/flipt':
        build_failure = build_failure or bool(re.search(
            r'^--- FAIL: Test\S+ \([\d.]+s\)$', logs, re.MULTILINE))
        # Some benchmark solutions explicitly update go.sum. Its missing
        # checksum entries in the unpatched tree are a task-manifest failure,
        # distinct from an unavailable external module (checked above).
        build_failure = build_failure or (
            'missing go.sum entry for module providing package' in logs
            and bool(re.search(r'^diff --git a/go.sum b/go.sum(?:\r?\n|$)',row.get('patch',''),re.MULTILINE)))
    return score_official_output(row, output, task_build_failed=build_failure)


def score_official_output(row, output, *, task_build_failed=False):
    """Apply the official set predicate, distinguishing known dynamic test names."""
    if not isinstance(output, dict) or "tests" not in output:
        raise RuntimeError("Official SWE evaluator returned no valid test results")
    if not isinstance(output["tests"], list) or any(
        not isinstance(t, dict) or not isinstance(t.get("name"), str)
        or not isinstance(t.get("status"), str) for t in output["tests"]
    ):
        raise RuntimeError("Official SWE evaluator returned malformed test results")
    statuses = {}
    for test in output["tests"]:
        name, status = test["name"], test["status"]
        if status not in {"PASSED", "FAILED", "SKIPPED", "ERROR"}:
            raise RuntimeError("Official SWE evaluator returned an unknown test status")
        if name in statuses and statuses[name] != status:
            raise RuntimeError("Official SWE evaluator returned conflicting statuses for " + name)
        statuses[name] = status
    reported = set(statuses)
    passed = {name for name, status in statuses.items() if status == "PASSED"}
    parse = lambda value: ast.literal_eval(value) if isinstance(value, str) else value
    required = set(parse(row["fail_to_pass"])) | set(parse(row["pass_to_pass"]))
    if not required:
        raise RuntimeError("No required tests in SWE instance")
    missing = required - reported
    if not reported and not task_build_failed:
        raise RuntimeError('Official grader produced no tests or task compilation evidence; technical NOSCORE')
    return {"score": float(required <= passed), "required_tests": len(required), "passed_required_tests": len(required & passed),
        "failed_required_tests": sorted(required - passed), "absent_required_tests": sorted(missing),
        "task_implementation_build_failed":task_build_failed,"valid_evaluation": True}


def _archive_files(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, payload, executable in files:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o755 if executable else 0o644
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def generate_patch(row, candidate: Path, model: str, artifacts: Path, ledger, run_id, role):
    import docker
    client = docker.from_env(timeout=2400)
    image = "jefzda/sweap-images:" + row["dockerhub_tag"]
    try:
        client.images.get(image)
    except docker.errors.ImageNotFound:
        client.images.pull(image)
    container = client.containers.run(image, entrypoint="/bin/sh", command=["-c", "sleep infinity"], detach=True,
        name="autoharness-swe-" + uuid.uuid4().hex[:12], mem_limit="8g", pids_limit=512)
    artifacts.mkdir(parents=True, exist_ok=True)
    try:
        # A clean base repository, without applying the reference patch/test patch.
        setup = container.exec_run(["/bin/sh", "-c", "pwd; git rev-parse --show-toplevel"])
        lines = setup.output.decode().strip().splitlines()
        if setup.exit_code or not lines or not lines[-1].startswith("/"):
            raise RuntimeError("Could not locate SWE repository")
        repo_dir = lines[-1]
        reset = container.exec_run(["git", "reset", "--hard", row["base_commit"]], workdir=repo_dir)
        if reset.exit_code:
            raise RuntimeError("Could not reset task repository to base commit")
        container.exec_run(["mkdir", "-p", "/opt/autoharness", "/tmp/autoharness-harness"])
        binary = ROOT / "third_party/opencode-1.19.5-linux/opencode"
        container.put_archive("/opt/autoharness", _archive_files([("opencode", binary.read_bytes(), True)]))
        config = json.loads((candidate / "opencode.json").read_text(encoding="utf-8"))
        config["instructions"] = ["/tmp/autoharness-harness/instructions.md"]
        config["model"] = model
        config.setdefault("provider", {})["opencode-go"] = {"options": {"apiKey": "{env:OPENCODE_GO_API_KEY}"}}
        if model.startswith("codex-acp/"):
            from .acp_gateway import provider_config, context_headers
            config["provider"].update(provider_config(model, headers=context_headers(run_id, role, "swe_bench_pro", row["instance_id"])))
        files = [("opencode.json", json.dumps(config).encode(), False)]
        files.extend((p.relative_to(candidate).as_posix(), p.read_bytes(), False) for p in candidate.rglob("*") if p.is_file() and p.name != "opencode.json" and ".git" not in p.parts)
        container.put_archive("/tmp/autoharness-harness", _archive_files(files))
        prompt = row["problem_statement"] + "\n\nRequirements:\n" + str(row.get("requirements", "")) + "\n\nInterface:\n" + str(row.get("interface", ""))
        key = os.environ.get("OPENCODE_GO_API_KEY")
        if not key and not model.startswith("codex-acp/"):
            raise RuntimeError("OPENCODE_GO_API_KEY is missing")
        result = container.exec_run(["/opt/autoharness/opencode", "run", "--format", "json", "--model", model, prompt],
            workdir=repo_dir, environment={"OPENCODE_CONFIG": "/tmp/autoharness-harness/opencode.json", "OPENCODE_CONFIG_DIR": "/tmp/autoharness-harness/.opencode", "OPENCODE_GO_API_KEY": key or "", "ACP_GATEWAY_TOKEN": os.environ.get("ACP_GATEWAY_TOKEN", ""), "OPENCODE_DISABLE_AUTOUPDATE": "true"})
        trace = result.output.decode("utf-8", errors="replace")
        (artifacts / "opencode.jsonl").write_text(trace, encoding="utf-8")
        record_cli_usage(trace, ledger, run_id, role, benchmark="swe_bench_pro", task_id=row["instance_id"])
        if result.exit_code:
            raise RuntimeError("OpenCode failed during SWE generation; inspect trace and reconcile usage")
        container.exec_run(["git", "add", "--intent-to-add", "."], workdir=repo_dir)
        diff = container.exec_run(["git", "diff", "--binary", row["base_commit"]], workdir=repo_dir)
        patch = diff.output.decode("utf-8")
        (artifacts / "patch.diff").write_text(patch, encoding="utf-8")
        return patch
    finally:
        container.remove(force=True)
        client.close()


def record_cli_usage(trace, ledger, run_id, role, **context):
    count = 0
    incomplete = False
    for index, line in enumerate(trace.splitlines()):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") != "step_finish":
            continue
        part = event.get("part", {})
        usage = normalize_opencode(part.get("tokens"))
        usage["provider_cost_usd"] = part.get("cost")
        request_id = str(part.get("id") or hashlib.sha256(line.encode()).hexdigest())
        ledger.record(request_id, run_id, role, usage, **context)
        incomplete = incomplete or not usage["usage_complete"]
        count += 1
    if not count:
        ledger.record("unknown:" + hashlib.sha256(trace.encode()).hexdigest(), run_id, role, normalize_opencode(None), **context)
        raise RuntimeError("No OpenCode step usage available; stop before more model work")
    if incomplete:
        raise RuntimeError("Incomplete OpenCode step usage; reconcile before more model work")
    return count


def smoke():
    manifest = json.loads(Path(__file__).with_name("manifests").joinpath("swe_bench_pro.json").read_text())
    row = load_row(manifest["train"][0])
    artifacts = ROOT / "results/autosaddler_reproduction/infrastructure/swe_v1_oracle"
    artifacts.mkdir(parents=True, exist_ok=True)
    gold = grade_patch(row, row["patch"], artifacts / "gold")
    empty = grade_patch(row, "", artifacts / "empty")
    result = {"task_id": row["instance_id"], "gold": gold, "empty": empty, "passed": gold["score"] == 1 and empty["score"] == 0, "model_requests": 0}
    (artifacts / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("SWE evaluator infrastructure check failed")


if __name__ == "__main__":
    smoke()
