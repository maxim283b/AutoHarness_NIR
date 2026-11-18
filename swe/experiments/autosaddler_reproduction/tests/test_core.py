import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.autosaddler_reproduction.accounting import UsageLedger, normalize_opencode
from experiments.autosaddler_reproduction.engine import BalancedSelection, MacroRanking
from experiments.autosaddler_reproduction.opencode_runtime import candidate_hash
from experiments.autosaddler_reproduction.runner import run_case

PACKAGE = Path(__file__).resolve().parents[1]


def test_native_opencode_visible_output_and_reasoning_recombine_once():
    usage = normalize_opencode({"input": 100, "output": 20, "reasoning": 5, "cache": {"read": 30, "write": 10}})
    assert usage["input_tokens"] == 140
    assert usage["output_tokens"] == 25
    assert usage["total_tokens"] == 165


def test_native_opencode_real_reasoning_usage_matches_transport_total():
    usage = normalize_opencode({"input": 23653, "output": 89, "reasoning": 10, "cache": {"read": 15360, "write": 0}, "total": 39112})
    assert usage["output_tokens"] == 99
    assert usage["total_tokens"] == 39112


def test_unknown_usage_remains_unknown(tmp_path):
    ledger = UsageLedger(tmp_path / "calls.sqlite")
    ledger.record("unknown", "run", "target_search", normalize_opencode(None))
    assert ledger.totals("run")["target_search"]["total_tokens"] is None
    ledger.close()


def test_acp_input_includes_cache_and_output_already_includes_reasoning():
    from experiments.autosaddler_reproduction.acp_checks import normalize_acp
    usage = normalize_acp({"usage": {"inputTokens": 100, "cachedReadTokens": 30, "cachedWriteTokens": 10, "outputTokens": 20, "thoughtTokens": 5, "totalTokens": 160}})
    assert usage["input_tokens"] == 140
    assert usage["total_tokens"] == 160
    assert not normalize_acp({})["usage_complete"]


def test_acp_permission_is_limited_to_known_task_mcp_call():
    from experiments.autosaddler_reproduction.acp_checks import allow_task_mcp_once
    params = {"_meta": {"is_mcp_tool_approval": True}, "toolCall": {"toolCallId": "call"}, "options": [{"optionId": "once", "kind": "allow_once"}]}
    assert allow_task_mcp_once(params, []) is None
    event = {"params": {"update": {"toolCallId": "call", "rawInput": {"server": "acp_smoke_task"}}}}
    assert allow_task_mcp_once(params, [event]) == "once"
    assert allow_task_mcp_once({**params, "_meta": {}}, [event]) is None
    event["params"]["update"]["rawInput"]["server"] = "another_server"
    assert allow_task_mcp_once(params, [event]) is None


def test_gateway_external_tool_decisions_are_validated():
    from experiments.autosaddler_reproduction.acp_gateway import parse_decision
    tools = [{"function": {"name": "read", "parameters": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}}}]
    response = parse_decision('{"content":null,"tool_calls":[{"name":"read","arguments":{"path":"/workspace/x"}}]}', tools)
    assert response["tool_calls"][0]["function"]["name"] == "read"
    with pytest.raises(ValueError, match="unavailable"):
        parse_decision('{"content":null,"tool_calls":[{"name":"native_shell","arguments":{}}]}', tools)
    from jsonschema import ValidationError
    with pytest.raises(ValidationError):
        parse_decision('{"content":null,"tool_calls":[{"name":"read","arguments":{}}]}', tools)
    assert parse_decision('{"content":"done","tool_calls":[]}', tools)["content"] == "done"


def test_gateway_images_are_forwarded_as_acp_images_not_base64_text():
    from experiments.autosaddler_reproduction.acp_gateway import conversation_blocks
    blocks = conversation_blocks({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAABBBB"}}]}]})
    assert blocks[1] == {"type": "image", "mimeType": "image/png", "data": "AAAABBBB"}
    assert "AAAABBBB" not in blocks[0]["text"]


def test_gateway_failed_start_removes_private_auth_and_preserves_original(tmp_path, monkeypatch):
    from experiments.autosaddler_reproduction import acp_gateway
    original = tmp_path / "account"
    original.mkdir()
    credential = '{"fixture":"not-a-real-credential"}'
    (original / "auth.json").write_text(credential)
    monkeypatch.setenv("CODEX_HOME", str(original))
    monkeypatch.setattr(acp_gateway, "ROOT", tmp_path)
    monkeypatch.setattr(acp_gateway.shutil, "which", lambda name: str(tmp_path / name))
    closed = []
    class FailedClient:
        def __init__(self, *args, **kwargs):
            pass
        def initialize(self):
            raise RuntimeError("startup failure")
        def close(self):
            closed.append(True)
    monkeypatch.setattr(acp_gateway, "ACPClient", FailedClient)
    gateway = acp_gateway.ACPGateway(tmp_path / "artifacts")
    with pytest.raises(RuntimeError, match="startup failure"):
        with gateway:
            pytest.fail("Failed startup must never enter the experiment")
    assert closed == [True]
    assert not (gateway.private_home / "auth.json").exists()
    assert (original / "auth.json").read_text() == credential


def test_case_selection_is_seeded_and_independent_of_manifest_order():
    from experiments.autosaddler_reproduction.case_selection import select_ids
    assert select_ids(["a", "b", "c"], 2, 42, "gaia2", "test") == select_ids(["c", "a", "b"], 2, 42, "gaia2", "test")
    assert select_ids(["c", "a", "b"], 2) == ["c", "a"]


def test_missing_cli_usage_is_saved_and_stops_execution(tmp_path):
    from experiments.autosaddler_reproduction.swe_adapter import record_cli_usage
    ledger = UsageLedger(tmp_path / "calls.sqlite")
    trace = json.dumps({"type": "step_finish", "part": {"id": "missing-usage"}})
    with pytest.raises(RuntimeError, match="Incomplete"):
        record_cli_usage(trace, ledger, "run", "target_search")
    assert ledger.totals("run")["target_search"]["total_tokens"] is None
    ledger.close()


def test_request_recording_is_idempotent_and_detects_drift(tmp_path):
    ledger = UsageLedger(tmp_path / "calls.sqlite")
    usage = normalize_opencode({"input": 3, "output": 2})
    assert ledger.record("r", "run", "optimizer", usage, phase="reflect")
    assert not ledger.record("r", "run", "optimizer", usage, phase="reflect")
    with pytest.raises(ValueError):
        ledger.record("r", "run", "optimizer", usage, phase="diagnose_patch")
    with pytest.raises(ValueError):
        ledger.record("r", "other-run", "optimizer", usage, phase="reflect")
    with pytest.raises(ValueError):
        ledger.record("r", "run", "target_search", usage, phase="reflect")
    assert ledger.totals("run")["optimizer"]["total_tokens"] == 5
    ledger.close()


def test_budget_reservation_prevents_overspend_and_keeps_unknown_cost(tmp_path):
    ledger = UsageLedger(tmp_path / "calls.sqlite")
    ledger.reserve("a", 3, 5)
    ledger.settle("a", None)
    with pytest.raises(RuntimeError):
        ledger.reserve("b", 3, 5)
    ledger.reserve("c", 2, 5)
    with pytest.raises(ValueError):
        ledger.reserve("c", 2, 5)
    ledger.close()


@pytest.mark.parametrize("tokens", [{"input": -1, "output": 2}, {"input": True, "output": 2}, {"input": 3, "output": 2, "total": 99}])
def test_invalid_usage_rejected(tokens):
    with pytest.raises(ValueError):
        normalize_opencode(tokens)


def test_cost_overrun_is_persisted_before_stop(tmp_path):
    ledger = UsageLedger(tmp_path / "calls.sqlite")
    ledger.reserve("a",3,5)
    with pytest.raises(RuntimeError):
        ledger.settle("a",4)
    with pytest.raises(RuntimeError):
        ledger.reserve("b",2,5)
    ledger.close()


def test_usage_ledger_accepts_judge_thread(tmp_path):
    import threading
    ledger = UsageLedger(tmp_path / "calls.sqlite")
    errors = []
    def record():
        try:
            ledger.record("judge-call","run","judge",normalize_opencode({"input":3,"output":2}))
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=record)
    thread.start();thread.join()
    assert not errors
    assert ledger.totals("run")["judge"]["total_tokens"] == 5
    ledger.close()


@pytest.mark.parametrize("benchmark,counts", [("gaia2", (75,65,300)), ("swe_bench_pro", (79,80,237)), ("terminal_bench_2", (30,19,40))])
def test_frozen_splits_disjoint_and_expected_size(benchmark, counts):
    manifest = json.loads((PACKAGE / "manifests" / f"{benchmark}.json").read_text())
    groups = [set(manifest[s]) for s in ("train", "development", "test")]
    assert tuple(map(len, groups)) == counts
    assert not (groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])


def test_balanced_training_batch():
    cases = [SimpleNamespace(case_id=f"{b}:{i}", payload={"benchmark": b}) for b in ("gaia2","swe_bench_pro","terminal_bench_2") for i in range(5)]
    policy = BalancedSelection(batch_size=6, seed=42)
    first, second = policy.select(cases, 0), policy.select(cases, 1)
    assert len(first.case_ids) == 6
    assert len(set(first.case_ids)) == 6
    assert first.case_ids != second.case_ids
    assert policy.select(cases, 0) == first
    with pytest.raises(ValueError):
        BalancedSelection(batch_size=4, seed=42)


def test_terminal_only_sampling_and_ranking():
    scope = ("terminal_bench_2",)
    cases = [SimpleNamespace(case_id=f"terminal_bench_2:{i}", payload={"benchmark":scope[0]}) for i in range(4)]
    policy = BalancedSelection(batch_size=2, seed=42, benchmarks=scope)
    first, second = policy.select(cases, 0), policy.select(cases, 1)
    assert len(first.case_ids) == 2
    assert set(first.case_ids).isdisjoint(second.case_ids)
    def evaluation(benchmark, value):
        return SimpleNamespace(observations=[SimpleNamespace(case_id=f"{benchmark}:a", score=value, is_valid=True)])
    ranking = MacroRanking(scope)
    candidates = [("H0", evaluation(scope[0], 0)), ("H1", evaluation(scope[0], 1))]
    assert ranking.select(candidates)[0] == "H1"
    with pytest.raises(ValueError, match="Incomplete"):
        ranking.select([("wrong", evaluation("gaia2", 1))])


@pytest.mark.parametrize("value", [[], ["typo"], ["terminal_bench_2"] * 2, "terminal_bench_2"])
def test_invalid_benchmark_scopes_are_rejected(value):
    from experiments.autosaddler_reproduction.benchmarks import selected_benchmarks
    with pytest.raises(ValueError):
        selected_benchmarks({"benchmarks": value})


def test_terminal_only_run_rejects_other_benchmarks_before_tools(tmp_path):
    case = SimpleNamespace(payload={"benchmark":"gaia2", "task_id":"unused"})
    with pytest.raises(ValueError, match="outside"):
        run_case(case, tmp_path, tmp_path, {"benchmarks":["terminal_bench_2"]}, "run", "target_search")


def test_terminal_configuration_does_not_require_gaia_judge(monkeypatch):
    from experiments.autosaddler_reproduction import doctor
    from autosaddler.v2.config.models import RunConfig
    config = RunConfig.load(PACKAGE / "configs/terminal_go_plus.draft.yaml")
    config.scenario.settings.pop("launch_ready")  # Exercise the separate judge/scope validation.
    observed = []
    def environment(benchmarks):
        observed.append(benchmarks)
        return {"checks": {"fixture": True}}
    monkeypatch.setattr(doctor, "inspect_environment", environment)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fixture-not-real")
    doctor.check_real_configuration(SimpleNamespace(config=config))
    assert observed == [("terminal_bench_2",)]


def test_terminal_draft_cannot_start_model_work():
    from experiments.autosaddler_reproduction import doctor
    from autosaddler.v2.config.models import RunConfig
    config = RunConfig.load(PACKAGE / "configs/terminal_go_plus.draft.yaml")
    with pytest.raises(RuntimeError, match="Draft launch is blocked"):
        doctor.check_real_configuration(SimpleNamespace(config=config))


def test_terminal_engine_and_final_resume_without_model_calls(tmp_path, monkeypatch):
    import yaml
    from experiments.autosaddler_reproduction.engine import build
    from experiments.autosaddler_reproduction import runner
    config = yaml.safe_load((PACKAGE / "configs/terminal_go_plus.draft.yaml").read_text())
    config["scenario"]["settings"]["mode"] = "smoke"
    config["scenario"]["settings"]["capability_phase_iterations"] = 1
    config["optimization"]["budget"]["max_iterations"] = 2
    config["provider"] = {"type": "experiment_smoke", "settings": {}, "capabilities": config["provider"]["capabilities"]}
    config["storage"]["run_root"] = str(tmp_path / "runs")
    path = tmp_path / "synthetic.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    runtime = build(path, "terminal-synthetic-" + tmp_path.name)
    assert {c.payload["benchmark"] for c in runtime.scenario.train_cases + runtime.scenario.development_cases} == {"terminal_bench_2"}
    result = runtime.engine.run()
    events = len(runtime.store.events())
    assert runtime.engine.run().selected_candidate_id == result.selected_candidate_id
    assert len(runtime.store.events()) == events
    # Exercise production final-selection/caching logic with a fake evaluator.
    # The synthetic run is confined to pytest's directory and never reported as performance.
    runtime.config.scenario.settings["mode"] = "real"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    calls = []
    def fake_case(case, candidate, artifacts, settings, run_id, role):
        calls.append((case, role))
        artifacts.mkdir(parents=True, exist_ok=True)
        return {"score": 1, "benchmark": case.payload["benchmark"], "model": settings["target_model"], "valid_evaluation": True}
    monkeypatch.setattr(runner, "run_case", fake_case)
    table = runner.evaluate_final(runtime, repetitions=1, n_test=2)
    assert len(table) == 1 and table[0]["benchmark"] == "terminal_bench_2"
    assert len(calls) == 4
    assert all(c.split == "test" and c.payload["benchmark"] == "terminal_bench_2" for c, role in calls)
    runner.evaluate_final(runtime, repetitions=1, n_test=2)
    assert len(calls) == 4, "Resume must not repeat completed H0/H1 executions"
    with pytest.raises(ValueError, match="frozen"):
        runner.evaluate_final(runtime, repetitions=1, n_test=3)


def test_macro_ranking_does_not_favor_larger_benchmark():
    def evaluation(scores):
        return SimpleNamespace(observations=[SimpleNamespace(case_id=f"{b}:{i}", score=v, is_valid=True) for b, values in scores.items() for i, v in enumerate(values)])
    one = ("one", evaluation({"gaia2":[1]*100,"swe_bench_pro":[0],"terminal_bench_2":[0]}))
    two = ("two", evaluation({"gaia2":[0]*100,"swe_bench_pro":[1],"terminal_bench_2":[1]}))
    assert MacroRanking().select([one,two])[0] == "two"


def test_test_tasks_cannot_enter_search(tmp_path):
    manifest = json.loads((PACKAGE / "manifests/terminal_bench_2.json").read_text())
    case = SimpleNamespace(split="test", payload={"benchmark":"terminal_bench_2", "task_id":manifest["test"][0]})
    with pytest.raises(ValueError, match="forbidden"):
        run_case(case, tmp_path, tmp_path, {}, "run", "target_search")


def test_candidate_identity_changes_with_content(tmp_path):
    p = tmp_path / "instructions.md"
    p.write_text("before")
    before = candidate_hash(tmp_path)
    p.write_text("after")
    assert candidate_hash(tmp_path) != before


def test_candidate_hash_tracks_tools_but_excludes_runtime_dependencies(tmp_path):
    (tmp_path / "instructions.md").write_text("fixed")
    before = candidate_hash(tmp_path)
    cache = tmp_path / ".opencode/node_modules/provider"
    cache.mkdir(parents=True)
    (cache / "index.js").write_text("generated dependency")
    assert candidate_hash(tmp_path) == before
    tools = tmp_path / ".opencode/tools"
    tools.mkdir()
    (tools / "helper.ts").write_text("export default {}")
    assert candidate_hash(tmp_path) != before


def test_template_has_no_models_or_credentials():
    config = json.loads((PACKAGE / "model_settings.example.json").read_text())
    assert config["target"]["model"] is None
    assert config["optimizer"]["model"] is None
