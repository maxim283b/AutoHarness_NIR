from data import load_splits


def test_disjoint_and_sizes():
    s = load_splits()
    ids = {k: {t["task_id"] for t in v} for k, v in s.items()}
    assert {k: len(v) for k, v in ids.items()} == {"train": 30, "val": 20, "test": 50}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])


def test_manifest_stable():
    assert [t["task_id"] for t in load_splits()["test"]] == [t["task_id"] for t in load_splits()["test"]]
