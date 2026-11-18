import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import policy_selection as ps  # noqa: E402


def unit(i, domain="python", split="search"):
    outcomes = {}
    for j, h in enumerate(ps.HARNESSES):
        success = float(j == (i % len(ps.HARNESSES)))
        outcomes[h] = {"success": success, "tokens": 100.0 + j, "tool_calls": 1.0,
                       "reward": success - .10 * (100 + j) / 1000 - .02}
    return {"key": [domain, str(i), 0], "domain": domain, "task_id": str(i), "seed": 0,
            "split": split, "task_template": "t", "features": {"n_words": i + 1},
            "outcomes": outcomes}


def test_frozen_rejects_validation_update():
    p = ps.Thompson(np.random.default_rng(0), 2)
    p.frozen = True
    with pytest.raises(AssertionError):
        p.update(0, .5, np.ones(2))


def test_fallback_charges_both_attempts_after_failure():
    u = unit(1)
    u["outcomes"]["direct"]["success"] = 0.0
    out, attempts = ps.fallback_outcome(u)
    assert attempts == ["direct", "verifier_loop"]
    assert out["tokens"] == u["outcomes"]["direct"]["tokens"] + u["outcomes"]["verifier_loop"]["tokens"]


def test_validation_is_score_only():
    train = [unit(i) for i in range(8)]
    validation = [unit(i + 100, split="validation") for i in range(4)]
    enc = ps.Encoder(train + validation, cross_domain=False)
    import io
    log = io.StringIO()
    result = ps.run_one(ps.LinUCB, train, validation, enc, train, 5, 0, "python", log)
    assert sum(result["final_state"]["n"]) == 5
    assert all(json.loads(x)["phase"] == "target_replay" for x in log.getvalue().splitlines())


def test_same_permutation_can_feed_every_policy():
    train = [unit(i) for i in range(10)]
    validation = [unit(i + 100, split="validation") for i in range(2)]
    enc = ps.Encoder(train + validation, cross_domain=False)
    import io
    seen = []
    for P in ps.POLICIES:
        log = io.StringIO()
        ps.run_one(P, train, validation, enc, train, 5, 3, "python", log)
        seen.append([json.loads(x)["task_key"] for x in log.getvalue().splitlines()])
    assert all(x == seen[0] for x in seen)
