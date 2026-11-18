import asyncio
from types import SimpleNamespace

import pytest

from experiments.autosaddler_reproduction.terminal_adapter import SharedOpenCode, OpenCode, trial_outcome


@pytest.mark.parametrize("reward", [0, 1])
def test_native_timeout_keeps_official_reward_after_confirmed_cleanup(reward):
    trial = {"exception_info":{"exception_type":"AgentTimeoutError"}, "verifier_result":{"rewards":{"reward":reward}}}
    result = trial_outcome(trial, cleanup_ok=True)
    assert result["valid_evaluation"] and result["agent_timed_out"]
    assert result["score"] == reward
    with pytest.raises(RuntimeError, match="cleanup"):
        trial_outcome(trial)


@pytest.mark.parametrize("error", ["VerifierTimeoutError", "RuntimeError", "NonZeroAgentExitCodeError"])
def test_other_harbor_errors_still_stop_the_run(error):
    with pytest.raises(RuntimeError, match="infrastructure"):
        trial_outcome({"exception_info":{"exception_type":error}, "verifier_result":{"rewards":{"reward":0}}}, cleanup_ok=True)


def test_timeout_without_verifier_result_is_not_a_score():
    with pytest.raises(RuntimeError, match="verifier"):
        trial_outcome({"exception_info":{"exception_type":"AgentTimeoutError"}}, cleanup_ok=True)


def test_cancellation_waits_for_cleanup_before_returning(monkeypatch):
    events = []
    async def run(self, instruction, environment, context):
        events.append("agent_started")
        await asyncio.sleep(5)
    monkeypatch.setattr(OpenCode, "run", run)
    class Environment:
        async def exec(self, **kwargs):
            assert kwargs["user"] == "root"
            await asyncio.sleep(0.02)
            events.append("cleanup_completed")
            return SimpleNamespace(return_code=0)
    async def verify():
        agent = object.__new__(SharedOpenCode)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(agent.run("fixture", Environment(), None), 0.02)
        events.append("verifier_can_start")
    asyncio.run(verify())
    assert events == ["agent_started", "cleanup_completed", "verifier_can_start"]


def test_cleanup_failure_prevents_normal_completion(monkeypatch):
    async def run(*args):
        return None
    monkeypatch.setattr(OpenCode, "run", run)
    class Environment:
        async def exec(self, **kwargs):
            return SimpleNamespace(return_code=1)
    with pytest.raises(RuntimeError, match="verification is forbidden"):
        asyncio.run(object.__new__(SharedOpenCode).run("fixture", Environment(), None))


def test_prior_budget_is_charged_on_restart(tmp_path, monkeypatch):
    import json
    from experiments.autosaddler_reproduction import go_gateway as go
    prior = tmp_path / "old.jsonl"
    prior.write_text(json.dumps({"id":"prior", "role":"target_search", "reservation":0.1,
        "usage":{"allowance":0.09,"total_tokens":123,"balance_cost":0}})+'\n')
    gateway = go.GoGateway(tmp_path / "new", allowance_limit=0.1, prior_journals=[prior])
    monkeypatch.setattr(go, "credential", lambda:pytest.fail("Budget must reject before network"))
    with pytest.raises(RuntimeError, match="budget"):
        gateway.forward("/anthropic/v1/messages", {"model":"qwen3.7-plus", "max_tokens":8192}, {"x-autoharness-role":"target_search"})
    assert gateway.accounting_summary()["roles"]["target_search"]["tokens"] == 123
    gateway.client.close()
    with pytest.raises(RuntimeError, match="history changed"):
        go.GoGateway(tmp_path / "new")
