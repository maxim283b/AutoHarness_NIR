import json

from pilot_common import (AppendOnlyJournal, HarnessConfig, ModelConfig,
                          OpenAICompatibleClient, PythonTaskAdapter, SQLTaskAdapter,
                          completed_pairs, run_task)


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return {"text": "SELECT 1", "prompt_tokens": 3, "completion_tokens": 2, "latency_s": 0.01}


def cfg(pattern="direct"):
    return HarnessConfig("Return SQL", pattern, max_repairs=1, memory_k=2,
                         model=ModelConfig(model="fake"), seed=7)


def test_trajectory_contains_steps_and_hashes():
    tr = run_task({"task_id": "q1", "prompt": "answer", "test_list": ["x"]}, cfg(),
                  FakeClient(), lambda answer: {"success": answer == "SELECT 1"})
    assert tr.success and tr.harness_config_hash and tr.harness_source_hash
    assert [s.kind for s in tr.steps] == ["model", "verifier"]
    assert tr.total_prompt_tokens == 3 and tr.model_calls == 1 and tr.tool_calls == 1


def test_verifier_loop_records_repair_and_tool_calls():
    class RepairClient(FakeClient):
        def chat(self, messages):
            self.calls.append(messages)
            text = "bad" if len(self.calls) == 1 else "good"
            return {"text": text, "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0}
    tr = run_task({"task_id": 2, "prompt": "answer", "test_list": ["x"]}, cfg("verifier_loop"),
                  RepairClient(), lambda answer: (answer == "good", "not good"))
    assert tr.success and tr.model_calls == 2 and tr.tool_calls == 2


def test_append_only_resume(tmp_path):
    j = AppendOnlyJournal(tmp_path / "trajectories.jsonl")
    tr = run_task({"task_id": 3, "prompt": "answer", "test_list": ["x"]}, cfg(),
                  FakeClient(), lambda answer: True)
    j.append_trajectory(tr)
    before = (tmp_path / "trajectories.jsonl").read_text()
    j.append_trajectory(tr)
    after = (tmp_path / "trajectories.jsonl").read_text()
    assert after.startswith(before) and len(list(j.records())) == 2
    assert ("3", "direct") in completed_pairs(j, config_hash=tr.harness_config_hash)


def test_config_is_json_serializable_and_openai_url():
    json.dumps(cfg().to_dict())
    c = OpenAICompatibleClient(ModelConfig(backend="openai_compatible", base_url="http://localhost:1234"))
    assert c.config.base_url.endswith("1234")


def test_memory_is_one_model_call_and_uses_examples():
    class Memory:
        def retrieve(self, prompt):
            return [{"input": "old", "output": "SELECT 1"}]
    client = FakeClient()
    tr = run_task({"task_id": "m1", "prompt": "answer", "test_list": ["x"]},
                  cfg("memory_assisted"), client, lambda answer: True, Memory())
    assert tr.model_calls == 1
    assert "Examples from memory" in client.calls[0][1]["content"]


def test_visible_test_flag_is_respected():
    client = FakeClient()
    config = HarnessConfig("Return SQL", "direct", use_visible_verifier=False,
                           model=ModelConfig(model="fake"))
    run_task({"task_id": "p1", "prompt": "answer", "test_list": ["SECRET_ASSERT"]},
             config, client, lambda answer: True, adapter=PythonTaskAdapter())
    assert "SECRET_ASSERT" not in str(client.calls)


def test_sql_adapter_never_exposes_reference_sql():
    task = {"task_id": "s1", "description": "Return one row", "schema": "CREATE TABLE t(x INT);",
            "allowed_operations": ["SELECT"], "features": {}, "template_group": "g1",
            "reference_sql": "SELECT TOP_SECRET FROM answers"}
    client = FakeClient()
    tr = run_task(task, cfg(), client, lambda answer: True, adapter=SQLTaskAdapter())
    assert "TOP_SECRET" not in str(client.calls)
    assert "reference_sql" not in tr.task_input
