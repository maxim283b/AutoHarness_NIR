from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "pilot" / "data"
TARGET = ROOT / "data"


def main() -> None:
    TARGET.mkdir(exist_ok=True)
    manifest = {"source": "google-research-datasets/mbpp sanitized; local pilot/data", "source_manifest": "pilot/splits_manifest.json", "visible_tests": "first public assert only", "hidden_tests": "remaining asserts, evaluator only", "selection": "first 12 train and first 20 test in pre-existing seeded split"}
    for split, count in (("train", 12), ("test", 20)):
        source = SOURCE / f"{split}.jsonl"
        lines = source.read_text(encoding="utf8").splitlines()[:count]
        target = TARGET / f"{split}.jsonl"
        target.write_text("\n".join(lines) + "\n", encoding="utf8")
        manifest[split] = {"count": count, "task_ids": [json.loads(line)["task_id"] for line in lines], "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (TARGET / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
