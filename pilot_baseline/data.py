"""MBPP-sanitized loading and fixed pilot splits (see pilot_selection.md, 3.3).

Usage:
    python data.py --out splits_manifest.json --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

DATASET = "google-research-datasets/mbpp"
CONFIG = "sanitized"
# our split -> (MBPP split, size)
SPLITS = {"train": ("train", 30), "val": ("validation", 20), "test": ("test", 50)}
HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "splits_manifest.json"
DATA_DIR = HERE / "data"


def _load_hf_split(hf_split: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(DATASET, CONFIG, split=hf_split)
    return [
        {
            "task_id": int(r["task_id"]),
            "prompt": r["prompt"],
            "test_imports": list(r["test_imports"]),
            "test_list": list(r["test_list"]),
            # reference solution is kept only for reference; it is never shown to the model
            "code": r["code"],
        }
        for r in ds
    ]


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def build_splits(seed: int = 0, out: Path = DEFAULT_MANIFEST) -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    manifest = {"dataset": DATASET, "config": CONFIG, "seed": seed, "splits": {}}
    for name, (hf_split, n) in SPLITS.items():
        rows = _load_hf_split(hf_split)
        rows.sort(key=lambda r: r["task_id"])
        rng = random.Random(seed)
        rng.shuffle(rows)
        chosen = rows[:n]
        path = DATA_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for r in chosen:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["splits"][name] = {
            "hf_split": hf_split,
            "n": len(chosen),
            "task_ids": [r["task_id"] for r in chosen],
            "file": str(path.relative_to(HERE)).replace("\\", "/"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest["manifest_sha256"] = _sha(manifest["splits"])
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_split(name: str, manifest_path: Path = DEFAULT_MANIFEST) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    info = manifest["splits"][name]
    path = HERE / info["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
        raise RuntimeError(f"{path} does not match splits_manifest.json; rebuild with data.py")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    m = build_splits(args.seed, args.out)
    for k, v in m["splits"].items():
        print(f"{k}: n={v['n']} from MBPP/{v['hf_split']}  sha256={v['sha256'][:12]}")
    print("manifest_sha256", m["manifest_sha256"])
