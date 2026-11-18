import json
import httpx
import pytest

from experiments.autosaddler_reproduction import go_gateway as go


def billing(amount=0.01, source="go-plus"):
    return {"cost":0, "opencode":{"billing_source":source, "allowance":amount, "tokens":{"input":100,"cache_read":20,"output":15,"reasoning":5}}}


def test_desktop_credential_export_accepts_array_and_selects_active_only():
    rows = [{"integrationID":"opencode", "active":False, "value":{"access":"old"}},
            {"integrationID":"opencode", "active":True, "value":{"access":"fixture"}}]
    assert go.exported_credential(json.dumps(rows))["access"] == "fixture"
    with pytest.raises(RuntimeError):
        go.exported_credential("[]")


def test_native_stream_merges_usage_without_double_counting():
    events = [{"type":"message_start", "message":{"usage":{"input_tokens":100}}},
              {"type":"message_delta", "usage":billing()}]
    raw = b"".join(("data: "+json.dumps(e)+"\n\n").encode() for e in events)
    result = go.metered_usage(go.usage_events(raw, True))
    assert result["total_tokens"] == 135
    assert result["allowance"] == 0.01


@pytest.mark.parametrize("usage", [billing(source="balance"), {**billing(),"cost":0.01}, {"cost":0}, billing(float("nan"))])
def test_unverified_or_paid_response_stops_work(usage):
    with pytest.raises(RuntimeError):
        go.metered_usage([{"usage":usage}])


def test_context_and_output_bounds():
    with pytest.raises(RuntimeError, match="context"):
        go.reservation({"messages":[{"content":"x"*(go.TEXT_REQUEST_BYTES+1)}]}, "qwen3.7-plus")
    with pytest.raises(RuntimeError, match="output"):
        go.reservation({"max_tokens":8193}, "kimi-k2.7-code")


def test_environment_only_agent_cannot_run_inference():
    import asyncio
    from experiments.autosaddler_reproduction.terminal_adapter import PrepareOnlyOpenCode
    with pytest.raises(RuntimeError, match="Inference is forbidden"):
        asyncio.run(PrepareOnlyOpenCode.run(None))


def test_unknown_attempt_reservation_survives_resume(tmp_path):
    gateway = go.GoGateway(tmp_path)
    gateway.append({"id":"a", "role":"target_search", "reservation":0.2, "status":"reserved"})
    gateway.client.close()
    resumed = go.GoGateway(tmp_path)
    assert resumed.spent() == 0.2
    resumed.append({"id":"a", "role":"target_search", "reservation":0.2, "status":"complete", "usage":{"allowance":0.03}})
    assert resumed.spent() == 0.03
    resumed.client.close()


def test_search_budget_cannot_consume_final_reserve(tmp_path, monkeypatch):
    gateway = go.GoGateway(tmp_path, allowance_limit=10, role_limits={"target_search":0.001})
    monkeypatch.setattr(go, "credential", lambda: pytest.fail("Budget must stop before auth or HTTP"))
    with pytest.raises(RuntimeError, match="role budget"):
        gateway.forward("/anthropic/v1/messages", {"model":"qwen3.7-plus", "max_tokens":100}, {"x-autoharness-role":"target_search"})
    assert not gateway.path.exists()
    gateway.client.close()


def test_quota_guard_requires_all_windows():
    value = {"usage":{name:{"status":"ok", "percent":0} for name in ("rolling","weekly","monthly")}}
    value["usage"]["rolling"]["percent"] = 70
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=value))) as client:
        with pytest.raises(RuntimeError, match="rolling"):
            go.quota(client, {"access":"fake", "metadata":{"orgID":"fake"}})


def test_search_request_cap_preserves_final_calls(tmp_path, monkeypatch):
    gateway = go.GoGateway(tmp_path, role_request_limits={"target_search":1})
    gateway.append({"id":"one", "role":"target_search", "reservation":0.01, "status":"reserved"})
    monkeypatch.setattr(go, "credential", lambda: pytest.fail("No HTTP after role request cap"))
    with pytest.raises(RuntimeError, match="role request budget"):
        gateway.forward("/anthropic/v1/messages", {"model":"qwen3.7-plus", "max_tokens":100}, {"x-autoharness-role":"target_search"})
    gateway.client.close()


def test_agent_api_error_cannot_become_zero_reward():
    from experiments.autosaddler_reproduction.terminal_adapter import check_agent_errors
    check_agent_errors('ordinary log\n{"type":"step_finish"}')
    with pytest.raises(RuntimeError, match="not a valid benchmark score"):
        check_agent_errors('{"type":"error","error":{"message":"quota exceeded"}}')


def test_real_launch_requires_matching_prepared_images(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import subprocess
    import yaml
    from experiments.autosaddler_reproduction import go_experiment as launcher, engine
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "PACKAGE", tmp_path)
    monkeypatch.setattr(engine, "prepare_seed", lambda:(tmp_path, "seed"))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"scenario":{"settings":{"benchmarks":["terminal_bench_2"],
        "train_per_benchmark":1, "development_per_benchmark":1, "case_selection_seed":42}}}))
    selected = {"train":["train-task"], "development":["dev-task"], "test":["test-task"]}
    go.save(tmp_path / "manifests/terminal_bench_2.json", selected)
    args = SimpleNamespace(config=config, n_test=1)
    with pytest.raises(FileNotFoundError):
        launcher.check_preflight(args)
    preflight = tmp_path / "results/autosaddler_reproduction/terminal_go_preflight"
    images = {task:{"tag":task+":pinned", "id":"image"} for tasks in selected.values() for task in tasks}
    go.save(preflight / "completed.json", {"selected":selected, "seed_revision":"seed", "images":images})
    for task in images:
        go.save(preflight / task / "installed.json", {"installed":True,"seed_revision":"seed","inference_calls":0})
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs:"image\n")
    assert launcher.check_preflight(args)["selected"] == selected
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs:"changed-image\n")
    with pytest.raises(RuntimeError, match="image changed"):
        launcher.check_preflight(args)


@pytest.mark.parametrize("trial_already_exists", [False, True])
def test_docker_preflight_retry_never_repeats_a_started_trial(tmp_path, monkeypatch, trial_already_exists):
    import subprocess
    from experiments.autosaddler_reproduction import terminal_adapter as adapter
    artifacts = tmp_path / "harbor"
    if trial_already_exists:
        (artifacts / "trial").mkdir(parents=True)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 1, "", "Docker daemon is not running. Please start Docker and try again.\n")
        go.save(artifacts / "trial/fixture/result.json", {"exception_info":None})
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(subprocess, "run", run)
    args = ("log-summary-date-ranges", adapter.ROOT / "experiments/autosaddler_reproduction/harness_seed",
        "opencode-go/qwen3.7-plus", artifacts, None, "fixture", "target_search")
    if trial_already_exists:
        with pytest.raises(RuntimeError, match="exactly one trial"):
            adapter.run_task(*args, install_only=True)
        assert len(calls) == 1
    else:
        assert adapter.run_task(*args, install_only=True)["installed"]
        assert len(calls) == 2


def test_forward_preserves_native_body_and_rejects_other_models(tmp_path, monkeypatch):
    gateway = go.GoGateway(tmp_path)
    seen = []
    def upstream(request):
        seen.append(request)
        return httpx.Response(200, json={"choices":[], "usage":billing(0.001)})
    gateway.client.close()
    gateway.client = httpx.Client(transport=httpx.MockTransport(upstream))
    monkeypatch.setattr(go, "credential", lambda:{"access":"fake", "metadata":{"orgID":"fake"}})
    monkeypatch.setattr(go, "quota", lambda *args:{"fixture":True})
    payload = {"model":"kimi-k2.7-code", "messages":[{"role":"user","content":"fixture"}],"max_tokens":100}
    status, headers, body = gateway.forward("/openai/v1/chat/completions", payload, {"x-autoharness-role":"optimizer"})
    assert status == 200 and json.loads(body)["usage"]["cost"] == 0
    assert json.loads(seen[0].content) == payload
    assert str(seen[0].url) == "https://opencode.ai/inference/go/openai/v1/chat/completions"
    assert gateway.spent() == 0.001
    with pytest.raises(ValueError, match="allowlist"):
        gateway.forward("/openai/v1/chat/completions", {"model":"unapproved"}, {})
    assert len(seen) == 1
    gateway.client.close()
