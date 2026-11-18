from experiments.harness_optimization.analysis import audit, bootstrap_tasks
from experiments.harness_optimization.engine import objective


def test_tail_objective_is_preregistered_lower_tail_not_tailrl():
    rows = [{"reward": x} for x in [1.0, 0.8, 0.6, 0.4, 0.2]]
    assert objective(rows, "mean") == 0.6
    assert objective(rows, "tail") == 0.4


def test_task_bootstrap_resamples_task_units():
    ci = bootstrap_tasks({"a": [0.0, 0.2], "b": [0.8, 1.0]}, n_boot=100)
    assert ci[0] == 0.5
    assert ci[1] <= ci[0] <= ci[2]


def test_validation_features_cannot_fit_encoder():
    # The preregistered core contains no learned encoder at all. This regression
    # test makes that design property explicit instead of silently fitting one.
    import experiments.harness_optimization.engine as engine
    assert not hasattr(engine, "fit_feature_encoder")


def test_audit_allows_preregistered_harness_token_budget_variation(tmp_path, monkeypatch):
    """Optimized decoding budget is a harness gene, not a model change."""
    import json
    from experiments.harness_optimization import analysis

    model = {
        "model_id": "fixed-model", "endpoint": "http://localhost/v1/chat/completions",
        "temperature": 0.0, "seed": 1, "context_budget": 2048,
        "generation_max_tokens": 256, "timeout_s": 180.0, "artifact_hash": "abc",
    }
    rows = []
    for index, max_tokens in enumerate((256, 384)):
        current = {**model, "generation_max_tokens": max_tokens}
        rows.append({
            "timestamp": "2026-01-01T00:00:00Z", "cache_key": f"k{index}",
            "candidate_hash": f"c{index}", "task_id": f"t{index}", "split": "search",
            "seed": 1, "model_config_hash": f"m{index}", "model": current,
            "harness": {"context_budget": 2048}, "success": 1.0, "reward": 0.98,
            "tokens": 0, "target_calls": 1, "proposer_calls": 0, "tool_calls": 1,
            "latency_s": 0.1, "infrastructure_error": None, "model_messages": [],
        })
    (tmp_path / "evaluations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (tmp_path / "optimization_log.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(analysis, "load_tasks", lambda: [])
    summary = {"logical_search_evaluations": 0, "logical_validation_evaluations": 0}
    report = audit(tmp_path, summary)
    assert "inconsistent model configuration" not in report["errors"]
