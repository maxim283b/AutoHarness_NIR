"""Create the checked-in SQLite fixture and frozen split manifest."""
import json, sqlite3
from pathlib import Path
from .tasks import DB_PATH, SCHEMA, dataset_hash, load_tasks, rows

def build():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # This is a generated fixture owned by this package; rebuild it atomically
    # from the fixed rows so repeated test setup stays deterministic.
    if DB_PATH.exists(): DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH); con.executescript(SCHEMA)
    for table, values in rows().items():
        con.executemany(f"INSERT INTO {table} VALUES ({','.join('?' for _ in values[0])})", values)
    con.commit(); con.close()
    tasks = load_tasks()
    split_names = ["search", "validation", "online", "held-out"]
    split = {name: [] for name in split_names}
    types = sorted({t["features"]["type"] for t in tasks})
    # Each type has four template groups (two broad query families, each split
    # into 3- and 2-variant subfamilies). Rotate their assignment so every
    # split receives every query type and exactly 15 tasks.
    for type_index, kind in enumerate(types):
        groups = sorted({t["template_group"] for t in tasks if t["features"]["type"] == kind})
        if len(groups) != 4:
            raise AssertionError(f"expected four template groups for {kind}, got {groups}")
        for group_index, group in enumerate(groups):
            destination = split_names[(group_index + type_index) % len(split_names)]
            split[destination].extend(t["task_id"] for t in tasks if t["template_group"] == group)
    for ids in split.values():
        ids.sort()
    first_experiment_ids = split["search"] + split["validation"] + split["online"][:10]
    manifest = {"version": 2, "seed": 20261008, "dataset_hash": dataset_hash(), "n_tasks": 60,
                "first_experiment_tasks": 40, "first_experiment_ids": first_experiment_ids, "splits": split,
                "family_policy": "template groups stay in one split; every query type occurs in every split",
                "held_out_policy": "read only for final evaluation; never used for updates"}
    (DB_PATH.parent.parent / "splits_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__": build()
