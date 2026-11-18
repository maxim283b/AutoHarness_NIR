"""MBPP-sanitized loading and the fixed pilot splits (30 train / 20 val / 50 test).

Usage: python data.py --out splits_manifest.json --seed 0
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
CACHE = HERE / "data" / "mbpp_sanitized.json"
SPLIT_SIZES = {"train": ("train", 30), "val": ("validation", 20), "test": ("test", 50)}


def _download() -> dict[str, list[dict]]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "sanitized")
    out = {}
    for split in ("train", "validation", "test", "prompt"):
        out[split] = [
            {
                "task_id": int(r["task_id"]),
                "prompt": r["prompt"],
                "code": r["code"],
                "test_imports": list(r["test_imports"]),
                "test_list": list(r["test_list"]),
            }
            for r in ds[split]
        ]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def load_raw() -> dict[str, list[dict]]:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return _download()


def make_splits(seed: int = 0) -> dict[str, list[dict]]:
    raw = load_raw()
    splits = {}
    for name, (src, n) in SPLIT_SIZES.items():
        rows = sorted(raw[src], key=lambda r: r["task_id"])
        random.Random(seed).shuffle(rows)
        splits[name] = rows[:n]
    return splits


def load_splits(manifest_path: Path = HERE / "splits_manifest.json") -> dict[str, list[dict]]:
    """Load splits and verify they match the frozen manifest."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    splits = make_splits(manifest["seed"])
    for name, rows in splits.items():
        ids = [r["task_id"] for r in rows]
        if ids != manifest["splits"][name]:
            raise RuntimeError(f"split {name} does not match manifest")
    if sha256_file(CACHE) != manifest["data_sha256"]:
        raise RuntimeError("data file hash does not match manifest")
    return splits


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "splits_manifest.json"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    splits = make_splits(a.seed)
    manifest = {
        "dataset": "google-research-datasets/mbpp:sanitized",
        "seed": a.seed,
        "data_sha256": sha256_file(CACHE),
        "splits": {k: [r["task_id"] for r in v] for k, v in splits.items()},
    }
    Path(a.out).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in manifest["splits"].items()})
