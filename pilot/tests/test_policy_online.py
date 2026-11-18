import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import policy_online as po  # noqa: E402


def test_new_online_stream_is_disjoint_and_not_heldout():
    tasks, manifest = po.new_online_tasks()
    ids = [x["task_id"] for x in tasks]
    assert len(ids) == len(set(ids)) == 5
    assert not set(ids) & set(manifest["first_experiment_ids"])
    assert not set(ids) & set(manifest["splits"]["held-out"])


def test_frozen_finalists_match_implementation():
    hashes = po.validate_frozen_configs()
    assert len(hashes) == 2
