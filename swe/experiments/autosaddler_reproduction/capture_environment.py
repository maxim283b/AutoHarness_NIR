"""Capture exact Python distributions and pinned upstream sources without secrets."""
from pathlib import Path
import subprocess
import json
import hashlib


def main():
    root = Path(__file__).resolve().parents[2]
    package = Path(__file__).parent
    code = "import importlib.metadata as m,json; print(json.dumps(sorted([(d.metadata['Name'],d.version) for d in m.distributions()])))"
    for env, name in ((".venv-experiment","requirements-experiment.lock.txt"),("third_party/Meta-ARE/.venv","requirements-are.lock.txt")):
        distributions = json.loads(subprocess.check_output([str(root/env/"Scripts/python.exe"),"-c",code],text=True,encoding="utf-8",timeout=30))
        lines = []
        for distribution, version in distributions:
            if distribution.lower() == "autosaddler":
                lines.append("-e ./third_party/AutoSaddler")
            elif distribution.lower() == "meta-agents-research-environments":
                lines.append("-e ./third_party/Meta-ARE")
            else:
                lines.append(f"{distribution}=={version}")
        (package/name).write_text("\n".join(lines)+"\n",encoding="utf-8")
    repos = {}
    for name in ("AutoSaddler","Meta-ARE","SWE-bench_Pro-os","terminal-bench-2"):
        path = root/"third_party"/name
        repos[name] = {"commit":subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip(),
                       "tracked_changes":subprocess.check_output(["git","-C",str(path),"status","--short","--untracked-files=no"],text=True).strip()}
        if name == "AutoSaddler":
            local_patch = subprocess.check_output(["git","-C",str(path),"diff","--binary","HEAD"])
            (package/"autosaddler_local.patch").write_bytes(local_patch)
            repos[name]["local_patch_sha256"] = hashlib.sha256(local_patch).hexdigest()
    base = "395d1dd512add1e3aeb5a6a092490768b51e3ce5"
    patch = subprocess.check_output(["git","-C",str(root/"third_party/Meta-ARE"),"diff","--binary",base])
    (package/"meta_are_windows.patch").write_bytes(patch)
    lock = {"platform":"Windows x64 + Docker Desktop Linux","python":"3.12.15","opencode":"1.19.5","repos":repos,
        "meta_are_base":base,"meta_are_patch_sha256":hashlib.sha256(patch).hexdigest(),
        "experiment_dependencies_sha256":hashlib.sha256((package/"requirements-experiment.lock.txt").read_bytes()).hexdigest(),
        "are_dependencies_sha256":hashlib.sha256((package/"requirements-are.lock.txt").read_bytes()).hexdigest()}
    lock["binary_sha256"] = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ("third_party/opencode-1.19.5/opencode.exe","third_party/opencode-1.19.5-linux/opencode")}
    adapter = root/"third_party/codex-acp/node_modules/@agentclientprotocol/codex-acp/dist/index.js"
    lock["codex_acp"] = {"version":"2.5.0", "adapter_sha256":hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "local_patch":"codex_acp_scope.patch", "note":"Preserve disabled global MCP servers when adding task-scoped action-selection tools"}
    (package/"environment_lock.json").write_text(json.dumps(lock,indent=2),encoding="utf-8")
    print(json.dumps(lock,indent=2))


if __name__ == "__main__":
    main()
