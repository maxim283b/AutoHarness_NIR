"""Run AutoSaddler V2 (local Ollama proposer + target) on MBPP, then evaluate on held-out test.

    python -m harness_lab.run_autosaddler --run-id as_default_s0 --policy default
    python -m harness_lab.run_autosaddler --run-id as_lite_s0    --policy lite

policy=default : stock AutoSaddler (strict improvement, any-component patches, verbose evidence)
policy=lite    : low-resource variant (net-wins>=2 acceptance, single-component patches,
                 failures-only compact evidence, failure-first batches)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from harness_lab import _paths
from harness_lab._paths import RESULTS

import yaml
from autosaddler.v2.config.registry import ScenarioRegistration, build_runtime, default_registry
from autosaddler.v2.plugins.api import SCENARIO_PLUGIN_API_VERSION, ScenarioPlugin

from harness_lab import policies
from harness_lab.benchmarks import load_humaneval_split, load_mbpp
from harness_lab.code_harness import SEED_COMPONENTS
from harness_lab.code_scenario import build_code_components
from harness_lab.evaluate import evaluate, summarize
from harness_lab.llm import LLMSettings
from harness_lab.ollama_provider import make_provider
from harness_lab.stats import paired_compare

POLICIES = {
    "default": dict(acceptance="matched_valid_strict_improvement", selection="fixed",
                    patch_scope="any", evidence={"max_chars": 12000, "failures_only": False}),
    "lite": dict(acceptance="net_wins_2", selection="failure_first",
                 patch_scope="single_component", evidence={"max_chars": 3500, "failures_only": True}),
}


def make_registry():
    registry = default_registry()
    registry.register_scenario(ScenarioRegistration(
        plugin=ScenarioPlugin(name="code_harness", api_version=SCENARIO_PLUGIN_API_VERSION,
                              factory=build_code_components), source="builtin"))
    registry.providers["ollama"] = make_provider
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "autosaddler_full"))
    try:  # optional local provider, autosaddler_full/ is not part of the repo
        from claude_bridge import make_provider as make_bridge_provider
        registry.providers["claude_bridge"] = make_bridge_provider
    except ImportError:
        pass
    policies.register(registry)
    policies.register_selection(registry)
    return registry


def build_config(args, run_root: Path) -> dict:
    pol = POLICIES[args.policy]
    rng = random.Random(args.data_seed)
    loader = load_humaneval_split if args.bench == "humaneval" else load_mbpp
    train = [t["task_id"] for t in loader("train")]
    dev = [t["task_id"] for t in loader("val")]
    rng.shuffle(train)
    rng.shuffle(dev)
    return {
        "schema_version": "autosaddler/v2",
        "scenario": {"type": "code_harness", "settings": {
            "baseline": SEED_COMPONENTS,
            "train_case_ids": train[: args.n_train],
            "development_case_ids": dev[: args.n_dev],
            "target": {"model": args.model, "base_url": args.base_url, "temperature": 0.0,
                       "seed": 0, "num_ctx": 4096, "max_tokens": 512,
                       "api_key_env": getattr(args, "api_key_env", "")},
            "evidence": pol["evidence"],
            "patch_scope": pol["patch_scope"],
        }},
        "provider": ({"type": "claude_bridge", "capabilities": ["read_workspace", "edit_workspace", "load_skills"],
                      "settings": {"bridge_dir": str(args.bridge_dir)}}
                     if args.proposer == "claude_bridge" else
                     {"type": "ollama", "capabilities": ["read_workspace", "edit_workspace", "load_skills"],
                      "settings": {"model": args.proposer_model or args.model,
                                   "base_url": getattr(args, "proposer_base_url", None) or args.base_url,
                                   "temperature": args.proposer_temperature, "seed": args.data_seed,
                                   "num_ctx": 8192, "max_tokens": 1024,
                                   "api_key_env": getattr(args, "proposer_api_key_env", "")}}),
        "optimization": {
            "task_selection": {"type": pol["selection"], "batch_size": args.batch_size, "seed": args.data_seed},
            "acceptance": {"type": pol["acceptance"]},
            "development": {"type": "full_on_accept"},
            "ranking": {"type": "mean_development_score"},
            "budget": {"max_iterations": args.iterations, "max_rollouts": args.max_rollouts},
            "diagnosis_patch_timeout_seconds": 3600.0 if args.proposer == "claude_bridge" else 600.0,
            "selection_timeout_seconds": 3600.0 if args.proposer == "claude_bridge" else 600.0,
            "reflection_timeout_seconds": 3600.0 if args.proposer == "claude_bridge" else 600.0,
            "session_retries": 2,
            "session_retry_backoff_seconds": 0.0,
        },
        "storage": {"type": "local", "run_root": str(run_root)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--bench", choices=["mbpp", "humaneval"], default="humaneval")
    ap.add_argument("--policy", choices=POLICIES, default="default")
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-dev", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--max-rollouts", type=int, default=400)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--proposer", choices=["ollama", "claude_bridge"], default="ollama",
                    help="claude_bridge: sessions are answered by the agent through autosaddler_full/claude_bridge.py")
    ap.add_argument("--bridge-dir", type=Path, default=RESULTS.parent / "autosaddler_full" / "bridge")
    ap.add_argument("--proposer-model")
    ap.add_argument("--proposer-temperature", type=float, default=0.3)
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--proposer-base-url", help="defaults to --base-url")
    ap.add_argument("--api-key-env", default="", help="env var with the target model's API key")
    ap.add_argument("--proposer-api-key-env", default="")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    run_root = RESULTS / "autosaddler"
    cfg = build_config(args, run_root)
    cfg_path = run_root / f"{args.run_id}.yaml"
    run_root.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    started = time.time()
    runtime = build_runtime(cfg_path, run_id=args.run_id, registry=make_registry())
    result = runtime.engine.run()
    opt_wall = time.time() - started
    run_dir = run_root / args.run_id
    selected = result.selected_candidate_id
    comp_path = run_dir / "candidates" / selected.removeprefix("sha256:") / "candidate.json"
    components = json.loads(comp_path.read_text(encoding="utf-8"))
    summary = {"run_id": args.run_id, "policy": args.policy, "args": vars(args),
               "selected_candidate_id": selected, "development_score": result.development_score,
               "iterations": result.iterations, "optimization_wall_s": opt_wall,
               "selected_is_seed": components == SEED_COMPONENTS, "components": components}
    ms = run_dir / "metrics-summary.json"
    if ms.exists():
        summary["metrics_summary"] = json.loads(ms.read_text(encoding="utf-8"))
    if not args.skip_test:
        settings = LLMSettings(model=args.model, base_url=args.base_url)
        loader = load_humaneval_split if args.bench == "humaneval" else load_mbpp
        test = loader("test")
        repair = args.bench != "humaneval"
        cache = RESULTS / "cache" / f"{args.bench}.jsonl"
        seed_rows = evaluate(dict(SEED_COMPONENTS), test, settings, use_repair=repair, cache=cache)
        opt_rows = evaluate(components, test, settings, use_repair=repair, cache=cache)
        summary["test_seed"] = summarize(seed_rows)
        summary["test_selected"] = summarize(opt_rows)
        summary["test_paired"] = paired_compare([r["reward"] for r in seed_rows], [r["reward"] for r in opt_rows])
    (run_dir / "summary_lab.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str),
                                              encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("components", "args", "metrics_summary")}, indent=1))


if __name__ == "__main__":
    main()
