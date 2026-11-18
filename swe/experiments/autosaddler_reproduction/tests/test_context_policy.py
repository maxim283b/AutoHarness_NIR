import json
import pytest
from experiments.autosaddler_reproduction import context_policy as policy
from experiments.autosaddler_reproduction import go_gateway as go


def test_native_policy_keeps_models_and_request_limits(monkeypatch):
    monkeypatch.setenv("GO_GATEWAY_URL", "http://fixture")
    for model in ("qwen3.7-plus", "kimi-k2.7-code"):
        name = "opencode-go/" + model
        config = {"provider":go.provider_config(name)}
        policy.configure(config, name, "file:///fixture.mjs")
        assert config["agent"]["compaction"]["model"] == name
        limits = config["provider"]["opencode-go"]["models"][model]["limit"]
        assert limits == {"context":128000,"output":8192,"input":32000}
        with pytest.raises(RuntimeError, match="context"):
            go.reservation({"messages":[{"content":"x"*(go.TEXT_REQUEST_BYTES+1)}]}, model)


def test_missing_compaction_tag_fails_before_inference(tmp_path, monkeypatch):
    gateway = go.GoGateway(tmp_path, require_context_policy=True)
    monkeypatch.setattr(go, "credential", lambda:pytest.fail("No network before policy tag check"))
    with pytest.raises(ValueError, match="accounting hook"):
        gateway.forward("/anthropic/v1/messages", {"model":"qwen3.7-plus"}, {"x-autoharness-role":"target_search"})
    gateway.client.close()


def test_compaction_is_subset_of_parent_role_budget_and_history(tmp_path):
    gateway = go.GoGateway(tmp_path)
    for purpose, cost in [("primary", 0.1), ("compaction", 0.05)]:
        gateway.append({"id":purpose,"role":"target_search","benchmark":"terminal_bench_2","purpose":purpose,
            "reservation":0.2,"usage":{"allowance":cost,"total_tokens":40,"balance_cost":0}})
    gateway.client.close()
    resumed = go.GoGateway(tmp_path)
    assert resumed.spent("target_search") == pytest.approx(0.15)
    summary = resumed.accounting_summary()
    assert summary["roles"]["target_search"]["tokens"] == 80
    assert sum(x["known_tokens"] for x in summary["by_benchmark_role_purpose"]) == 80
    assert len(summary["by_benchmark_role_purpose"]) == 2
    resumed.client.close()
