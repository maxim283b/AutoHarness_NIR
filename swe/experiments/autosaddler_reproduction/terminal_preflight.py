"""Prepare selected task containers without agent inference or test outcomes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tomllib

from .go_gateway import save, ROOT
from .case_selection import select_ids


def main():
    from .engine import prepare_seed
    from .terminal_adapter import run_task
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).parent / "manifests/terminal_bench_2.json").read_text())
    repo, revision = prepare_seed()
    artifacts = ROOT / "results/autosaddler_reproduction/terminal_go_preflight"
    selected = {split:select_ids(manifest[split], n, 42, "terminal_bench_2", split) for split,n in (("train",4),("development",3),("test",5))}
    save(artifacts / "task_selection.json", selected)
    jobs = [(split, task_id) for split, tasks in selected.items() for task_id in tasks]
    for split, task_id in jobs:
        task = ROOT / "third_party/terminal-bench-2" / task_id
        data = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
        if data["environment"].get("gpus", 0):
            raise RuntimeError("Selected task requires unprovisioned GPU: " + task_id)
        if data["environment"].get("memory_mb", 0) > 4096:
            raise RuntimeError("Task exceeds the two-container preparation resource allowance")
        if not (task / "tests/test.sh").is_file():
            raise RuntimeError("Missing verifier entrypoint: " + task_id)
    def prepare(job):
        split, task_id = job
        target = artifacts / task_id
        if not (target / "installed.json").exists():
            print("Prepare environment only: " + split + " / " + task_id, flush=True)
            number = len(list(target.glob("attempt-*")))+1
            result = run_task(task_id, repo, "opencode-go/qwen3.7-plus", target / ("attempt-"+str(number)), None, "environment-only", "target_search", install_only=True)
            result.update(seed_revision=revision, inference_calls=0)
            save(target / "installed.json", result)
    if args.install:
        # Only setup is parallel: at most 8 GiB on this 16 GiB Docker host.
        # Agent evaluations remain strictly sequential and unchanged.
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(prepare, jobs))
    if args.install:
        images = {}
        for tasks in selected.values():
            for task_id in tasks:
                data = tomllib.loads((ROOT / "third_party/terminal-bench-2" / task_id / "task.toml").read_text(encoding="utf-8"))
                tag = data["environment"]["docker_image"]
                inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", tag], encoding="utf-8", timeout=30))[0]
                images[task_id] = {"tag":tag, "id":inspected["Id"], "digests":inspected.get("RepoDigests", [])}
        save(artifacts / "completed.json", {"selected":selected, "seed_revision":revision, "images":images, "inference_calls":0})
    print("Selected task environments checked; no model or verifier calls made", flush=True)


if __name__ == "__main__":
    main()
