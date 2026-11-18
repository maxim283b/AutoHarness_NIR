"""Freeze a new 24/16/40 split from the already certified 80 SWE V1 tasks.

The prior campaign's three completed test IDs are excluded from this run's
held-out test. Selection uses only their exposure status, never their scores.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parent
OUT = Path(__file__).resolve().parent
SEED = 20261112


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, content: str) -> None:
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"Frozen plan drift: {path}")
    path.write_text(content)


def main() -> None:
    original_path = ROOT / "fullsource/selected-manifest.json"
    original = json.loads(original_path.read_text())
    partial_path = MAIN / "experiments/swe_opencode_20261108/data/partial-test-results.csv"
    with partial_path.open(newline="", encoding="utf-8-sig") as stream:
        exposed = {row["task_id"] for row in csv.DictReader(stream)}
    source_test = original["test"]
    if len(exposed) != 3 or not exposed <= set(source_test):
        raise RuntimeError("Unexpected prior held-out exposure")
    order = lambda ids, label: sorted(ids, key=lambda task: hashlib.sha256(f"{SEED}:{label}:{task}".encode()).hexdigest())
    test = order(set(source_test) - exposed, "test")[:40]
    remaining = set(original["train"] + original["development"] + source_test) - set(test)
    train = order(remaining, "train")[:24]
    development = order(remaining - set(train), "development")[:16]
    if (len(train), len(development), len(test)) != (24, 16, 40) or len(set(train + development + test)) != 80:
        raise RuntimeError("Split is not exactly 24/16/40 without overlap")
    execution_order = sorted(test, key=lambda task: hashlib.sha256(
        f"{SEED}:swe_bench_pro:test:{task}".encode()).hexdigest())
    plan = {"benchmark": "SWE-bench Pro V1", "source_manifest_sha256": sha(original_path),
            "source_jsonl_sha256": original["raw_sha256"], "selection_seed": SEED,
            "selection_rule": "test from unexposed prior test IDs by SHA-256; remaining IDs split 24/16 by SHA-256",
            "previously_exposed_test_ids": sorted(exposed), "train": train,
            "development": development, "test": test,
            "test_fallback_first_16": execution_order[:16],
            "prior_observations_disclosure": "Previous campaign evaluated 8 train, 8 development, and 3 test IDs; no prior results enter either optimizer's workspace."}
    write_once(OUT / "manifest.json", json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    source = yaml.safe_load((ROOT / "fullsource/config.frozen.yaml").read_text())
    source["optimization"]["budget"]["max_iterations"] = 6
    source["optimization"]["budget"]["max_rollouts"] = 384
    source["optimization"]["task_selection"]["batch_size"] = 8
    source["optimization"]["task_selection"]["seed"] = SEED
    source["provider"]["settings"]["model"] = "opencode-go/glm-5.3"
    settings = source["scenario"]["settings"]
    settings.update({"fullsource_manifest": "improved_v2/manifest.json", "train_per_benchmark": 24,
                     "development_per_benchmark": 16, "batch_per_benchmark": 8,
                     "capability_phase_iterations": 3, "case_selection_seed": SEED,
                     "optimizer_model_name": "GLM-5.3"})
    source["storage"]["run_root"] = str(ROOT / "results/autosaddler_reproduction/improved_v2_runs")
    for label in ("H1", "H2"):
        config = json.loads(json.dumps(source))
        config["scenario"]["settings"]["improved_autosaddler_v2"] = label == "H2"
        write_once(OUT / f"{label}.yaml", yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({"manifest_sha256": sha(OUT / "manifest.json"),
                      "H1_sha256": sha(OUT / "H1.yaml"), "H2_sha256": sha(OUT / "H2.yaml"),
                      "counts": {"train": 24, "development": 16, "test": 40}}))


if __name__ == "__main__":
    main()
