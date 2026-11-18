"""Provision pinned dataset manifests; never choose splits from measured scores."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = Path(__file__).with_name("manifests")
DATA = ROOT / "third_party/experiment_data"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    path = MANIFESTS / name
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"Frozen manifest changed: {path}; create an explicit new protocol version")
    path.write_text(payload, encoding="utf-8")


def terminal_bench():
    root = ROOT / "third_party/terminal-bench-2"
    tasks = sorted(p.name for p in root.iterdir() if (p / "task.toml").is_file())
    if len(tasks) != 89:
        raise ValueError(f"Expected the 89-task Terminal-Bench 2 corpus, got {len(tasks)}")
    random.Random(42).shuffle(tasks)
    save("terminal_bench_2.json", {
        "benchmark": "terminal-bench@2.0", "split_origin": "local seeded partition, NOT verified as the unpublished exact AutoSaddler IDs",
        "seed": 42, "source_revision": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "train": tasks[:30], "development": tasks[30:49], "test": tasks[49:],
        "task_hashes": {t: {str(p.relative_to(root / t)).replace('\\','/'): sha(p) for p in sorted((root/t).rglob('*')) if p.is_file()} for t in sorted(tasks)},
    })
    print("Terminal-Bench 2: 30/19/40, frozen local split", flush=True)


def gaia2():
    from autosaddler.v2.plugins.meta_are.scenario_provisioning import ScenarioProvisioningRequest, provision_gaia2_scenarios
    split_root = ROOT / "third_party/AutoSaddler/configs/datasets/GAIA2"
    files = [split_root / n for n in ("train.json", "val.json", "test_universe_21.json", "test_universe_22.json", "test_universe_27.json")]
    revision = "78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99"
    dest = (DATA / "gaia2").resolve()
    result = provision_gaia2_scenarios(ScenarioProvisioningRequest(destination_root=dest, source_revision=revision, manifests=tuple(files)))
    values = [json.loads(p.read_text()) for p in files]
    flatten = lambda d: [case for ids in d.values() for case in ids]
    save("gaia2.json", {"benchmark": "gaia2", "source_revision": revision,
        "train": flatten(values[0]), "development": flatten(values[1]),
        "test": sum((flatten(v) for v in values[2:]), []),
        "manifest_hashes": {p.name: sha(p) for p in files},
        "scenario_descriptor_sha256": sha(result.source_descriptor)})
    print(f"GAIA2: {result.file_count} scenarios provisioned", flush=True)


def swe():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    repo = "ScaleAI/SWE-bench_Pro"
    revision = HfApi().dataset_info(repo, revision="v1.0").sha
    # Explicit v1 is essential: 'default' now points to a different V2 benchmark.
    # At the immutable v1.0 tag the original corpus is named 'default'; the
    # later 'v1' config exists only on current main. The 731-row check guards it.
    rows = list(load_dataset(repo, name="default", split="test", revision=revision))
    if len(rows) != 731:
        raise ValueError(f"Expected SWE-Bench Pro V1's 731 tasks, got {len(rows)}")
    DATA.mkdir(parents=True, exist_ok=True)
    raw_path = DATA / "swe_bench_pro_v1.jsonl"
    raw_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    groups = {"train": {"qutebrowser/qutebrowser"}, "development": {"future-architect/vuls", "NodeBB/NodeBB"}, "test": {"ansible/ansible", "flipt-io/flipt", "element-hq/element-web", "vector-im/element-web"}}
    splits = {name: sorted(row["instance_id"] for row in rows if row.get("repo") in repos) for name, repos in groups.items()}
    dev = []
    for repo_name in sorted(groups["development"]):
        ids = sorted(row["instance_id"] for row in rows if row.get("repo") == repo_name)
        if len(ids) < 40:
            raise ValueError(f"Not enough development cases in {repo_name}")
        dev.extend(random.Random(42).sample(ids, 40))
    splits["development"] = sorted(dev)
    expected = {"train": 79, "development": 80, "test": 237}
    if {k: len(v) for k, v in splits.items()} != expected:
        # Surface upstream/repository-name drift; do not silently change groups.
        repos = sorted({str(row.get("repo")) for row in rows})
        raise ValueError(f"SWE split counts differ: { {k:len(v) for k,v in splits.items()} }; repos={repos}")
    save("swe_bench_pro.json", {"benchmark": "swe-bench-pro-v1", "source_revision": revision, "source_config": "default-at-v1.0", "split_origin": "paper repository groups and counts; dev IDs sampled with seed 42 because exact paper IDs are not supplied", "raw_sha256": sha(raw_path), **splits})
    print("SWE-Bench Pro V1: 79/80/237, frozen repository split", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=["gaia2", "swe", "terminal-bench"])
    args = parser.parse_args()
    {"gaia2": gaia2, "swe": swe, "terminal-bench": terminal_bench}[args.benchmark]()
