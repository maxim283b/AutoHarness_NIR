"""Append-only JSONL journal and resume helpers."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable


class AppendOnlyJournal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def append_trajectory(self, trajectory) -> None:
        """Append one complete trajectory as one immutable JSON object."""
        record = trajectory.to_dict() if hasattr(trajectory, "to_dict") else dict(trajectory)
        record["record_type"] = "trajectory"
        record.setdefault("pattern", record.get("harness", {}).get("pattern"))
        self.append(record)

    def records(self) -> Iterable[dict]:
        if not self.path.exists():
            return []
        def read():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        return read()


def completed_pairs(journal: AppendOnlyJournal, *, config_hash: str | None = None) -> set[tuple[str, str]]:
    done = set()
    for r in journal.records():
        if r.get("record_type", "trajectory") != "trajectory":
            continue
        if config_hash is not None and r.get("harness_config_hash") != config_hash:
            continue
        pattern = r.get("pattern") or r.get("harness", {}).get("pattern")
        if r.get("task_id") is not None and pattern:
            done.add((str(r["task_id"]), str(pattern)))
    return done
