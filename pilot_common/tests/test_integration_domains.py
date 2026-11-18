import pytest

from pilot_common import (HarnessConfig, ModelConfig, PythonTaskAdapter,
                          SQLTaskAdapter, run_task)


class SequenceClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def chat(self, messages):
        self.messages.append(messages)
        return {"text": next(self.responses), "prompt_tokens": 2,
                "completion_tokens": 1, "latency_s": 0.0}


class Memory:
    def retrieve(self, prompt):
        return [{"problem": "example", "solution": "good"}]


PY_TASK = {"task_id": "py-1", "prompt": "Return good", "test_list": ["VISIBLE", "HIDDEN_SECRET"], "test_imports": []}
SQL_TASK = {"task_id": "sql-1", "description": "Return good", "schema": "CREATE TABLE t(x INT);",
            "allowed_operations": ["SELECT"], "features": {"type": "filter"},
            "template_group": "filter-a", "reference_sql": "SELECT HIDDEN_SECRET FROM private"}


@pytest.mark.parametrize("task,adapter", [(PY_TASK, PythonTaskAdapter()), (SQL_TASK, SQLTaskAdapter())])
@pytest.mark.parametrize("pattern,responses,expected_models,expected_tools", [
    ("direct", ["good"], 1, 1),
    ("self_refine", ["draft", "good"], 2, 1),
    ("verifier_loop", ["bad", "good"], 2, 2),
    ("memory_assisted", ["good"], 1, 1),
])
def test_two_domains_four_patterns_no_private_leak(task, adapter, pattern, responses,
                                                   expected_models, expected_tools):
    client = SequenceClient(responses)
    config = HarnessConfig("Return only the answer", pattern, max_repairs=1, memory_k=2,
                           model=ModelConfig(backend="openai_compatible", model="fake",
                                             base_url="http://127.0.0.1:1234"))
    tr = run_task(task, config, client, lambda output: (output == "good", "not good"),
                  memory=Memory(), adapter=adapter)
    assert tr.success
    assert tr.model_calls == expected_models
    assert tr.tool_calls == expected_tools
    serialized_messages = str(client.messages)
    assert "HIDDEN_SECRET" not in serialized_messages
    assert "reference_sql" not in serialized_messages
    if pattern == "memory_assisted":
        assert "Examples from memory" in serialized_messages

