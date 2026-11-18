"""Exercise the actual ARE LiteLLM engine against the ACP-backed judge route."""
import argparse
import json
import os
from pathlib import Path


def main():
    from are.simulation.agents.llm.litellm import litellm_engine
    from .accounting import UsageLedger
    from .acp_gateway import context_headers, MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    original = litellm_engine.completion
    raw = []
    def completion(*a, **kw):
        kw["extra_headers"] = context_headers(args.run_id, "judge", "gaia2", "judge-route-probe")
        response = original(*a, **kw)
        raw.append(response)
        return response
    litellm_engine.completion = completion
    engine = litellm_engine.LiteLLMEngine(litellm_engine.LiteLLMModelConfig(model_name=MODEL, provider="openai",
        endpoint=os.environ["ACP_GATEWAY_LOCAL_URL"], api_key=os.environ["ACP_GATEWAY_TOKEN"]))
    answer = engine.chat_completion([{"role": "user", "content": 'Infrastructure route check. Reply with exactly {"route_ready":true}. Do not use tools.'}])
    if isinstance(answer, tuple):
        answer = answer[0]
    parsed = json.loads(answer.strip().removeprefix("```json").removesuffix("```").strip())
    if parsed != {"route_ready": True} or len(raw) != 1:
        raise RuntimeError("ARE judge route probe failed")
    usage = raw[0].usage
    root = Path(__file__).resolve().parents[2]
    ledger = UsageLedger(root / "results/autosaddler_reproduction/usage" / args.run_id / "requests.sqlite")
    ledger.record(raw[0].id, args.run_id, "judge", {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens, "usage_complete": True}, benchmark="gaia2", task_id="judge-route-probe", infrastructure_only=True)
    ledger.close()
    result = {"passed": True, "native_are_litellm_engine": True, "judge_model": MODEL, "response": parsed, "total_tokens": usage.total_tokens}
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
