from analysis.signal_analysis import analyze, winners


def synthetic_rows():
    rows = []
    for i in range(24):
        template = f"template-{i % 6}"
        for seed in (0, 1):
            for harness in ("direct", "verifier_loop"):
                # The best harness alternates by template, producing a known signal.
                best = (i % 2 == 0)
                success = int((harness == "direct") == best)
                rows.append({
                    "task_id": f"task-{i}", "task_template": template, "split": "search",
                    "seed": seed, "harness": harness, "model": "synthetic@sha256:test",
                    "success": success, "tokens": 100 if harness == "direct" else 200,
                    "tool_calls": 0 if harness == "direct" else 1, "latency_s": 1.0,
                    "task_features": {"even": int(i % 2 == 0), "index": i},
                    "trajectory": [{"step": "solve", "input_hash": "x", "output_hash": "y"}],
                })
    rows.append({**rows[0], "task_id": "heldout-1", "split": "heldout"})
    return rows


def test_analysis_excludes_heldout_from_fit():
    report = analyze(synthetic_rows(), seed=7)
    assert report["config"]["heldout_used_for_fit"] is False
    assert report["n_rows"] == 97
    assert report["n_rows_nonheldout"] == 96
    assert report["heldout_n_tasks"] == 1
    assert report["main"]["winners"]["n_task_seed_units"] == 48
    assert report["main"]["winners"]["mean_oracle_gap"] > 0


def test_multiple_seeds_are_independent_and_order_invariant():
    rows = [r for r in synthetic_rows() if r["split"] != "heldout"]
    a = winners(rows, .10, .02)
    b = winners(list(reversed(rows)), .10, .02)
    assert a["n_task_seed_units"] == 48
    assert a["mean_oracle_gap"] == b["mean_oracle_gap"]
    assert a["unique_best_share"] == b["unique_best_share"]


def test_heldout_does_not_change_fit_statistics():
    rows = synthetic_rows()
    base = analyze(rows, seed=7)
    extra = {**rows[0], "task_id": "heldout-extra", "split": "heldout", "success": 0}
    changed = analyze(rows + [extra], seed=7)
    assert base["main"] == changed["main"]
    assert base["predictive"] == changed["predictive"]
