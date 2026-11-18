import json

from data import DEFAULT_MANIFEST, load_split


def test_splits_disjoint_and_sized():
    m = json.loads(DEFAULT_MANIFEST.read_text())
    ids = {k: set(v["task_ids"]) for k, v in m["splits"].items()}
    assert {k: len(v) for k, v in ids.items()} == {"train": 30, "val": 20, "test": 50}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])


def test_load_split_matches_manifest():
    m = json.loads(DEFAULT_MANIFEST.read_text())
    for name in ("train", "val", "test"):
        rows = load_split(name)
        assert [r["task_id"] for r in rows] == m["splits"][name]["task_ids"]
        assert all(r["test_list"] for r in rows)
