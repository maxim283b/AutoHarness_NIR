import pytest
from pydantic import ValidationError

from src.controllers.policies import Decision, Fixed, Rule, LLMController, LinUCB, features
from src.execution.llm import Client, Limits
from src.execution.runner import execute_task
from src.execution.state import TaskState, PROTOCOLS
from src.evaluation.checker import run_tests


TASK = {"task_id": 1, "prompt": "Write function square(x) returning x*x.", "test_imports": [], "test_list": ["assert square(2) == 4", "assert square(3) == 9"]}


def mock_client():
    return Client("mock", "http://invalid", "", Limits(12, 10000, 30), mock=True)


@pytest.mark.parametrize("protocol,roles", [("P1", ["executor"]), ("P2", ["planner", "executor"]), ("P3", ["executor", "verifier"]), ("P4", ["planner", "executor", "verifier"])])
def test_fixed_protocol_roles(protocol, roles):
    result = execute_task(TASK, Fixed(protocol), mock_client(), max_attempts=1)
    assert [e["kind"] for e in result["events"] if e["kind"] != "visible_check"] == roles
    assert result["calls"] == len(roles)
    assert result["initial_protocol"] == protocol
    assert result["mock"] is True
    assert result["tool_calls"] == 1
    assert result["errors"] == 1


def test_switch_retains_code_feedback_and_event_history():
    result = execute_task(TASK, Rule(), mock_client())
    assert [(s["previous"], s["current"]) for s in result["switches"]] == [("P1", "P3"), ("P3", "P4")]
    first = result["switches"][0]["state_snapshot"]
    assert "def square" in first["code"]
    assert first["feedback"]
    assert first["event_count"] == 2
    assert result["switches"][1]["state_snapshot"]["event_count"] > first["event_count"]
    assert all(s["final_success"] is False for s in result["switches"])


def test_hidden_test_is_not_sent_to_model():
    result = execute_task(TASK, Fixed("P1"), mock_client(), max_attempts=1)
    assert "square(3)" not in str(result["events"])


def test_controller_schema_is_strict():
    with pytest.raises(ValidationError):
        Decision.model_validate({"protocol": "P5", "reason": "bad", "confidence": 2})
    with pytest.raises(ValidationError):
        Decision.model_validate({"protocol": "P1", "reason": "ok", "confidence": 0.5, "extra": 1})


def test_checker_and_bandit():
    assert run_tests("def square(x): return x*x", [], TASK["test_list"])[0]
    assert not run_tests("import os\ndef square(x): return 4", [], TASK["test_list"])[0]
    bandit = LinUCB()
    state = TaskState(task_id=1, prompt=TASK["prompt"], visible_test=TASK["test_list"][0], imports=[])
    x = features(state, 10000)
    for _ in range(5):
        bandit.update("P2", x, 1)
    assert bandit.choose(state, mock_client()).protocol in PROTOCOLS


def test_llm_controller_overhead_and_hard_call_limit():
    controlled = execute_task(TASK, LLMController(), mock_client(), max_attempts=1)
    assert controlled["controller_calls"] == 1
    assert controlled["controller_tokens"] > 0
    assert controlled["calls"] == controlled["controller_calls"] + 2  # P3 executor + reviewer
    limited = execute_task(TASK, Fixed("P4"), Client("mock", "http://invalid", "", Limits(1, 10000, 30), mock=True))
    assert limited["calls"] == 1
    assert "BudgetExceeded" in limited["error"]
