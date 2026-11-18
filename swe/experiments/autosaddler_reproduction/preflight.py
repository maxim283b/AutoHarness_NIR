"""Read-only OpenRouter metadata checks. Never sends an inference request.

Run from any directory: python experiments/autosaddler_reproduction/preflight.py
Use --online to fetch public model prices and sanitized current-key budget data.
This is an estimator, not a runtime spend guard.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    "anthropic/claude-opus-4.6",
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen3-coder",
    "deepseek/deepseek-v3.2",
)


def read_key() -> str | None:
    """Read only the two supported variable names; do not export .env contents."""
    names = ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY")
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and name.strip() in names:
                key = value.strip().strip("\"").strip("'")
                if key:
                    return key
    return None


def get_metadata(endpoint: str, key: str | None = None) -> dict:
    headers = {"User-Agent": "AutoHarness-Reproduction-Preflight"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = Request("https://openrouter.ai/api/v1/" + endpoint, headers=headers)
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    paper = protocol["paper_protocol"]
    key = read_key()
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference_requests_sent": 0,
        "key_present": bool(key),
        "budget_cap_usd": protocol["budget_usd"]["total_cap"],
        "paper_paired_test_rollouts_without_optimization": sum(
            paper[name]["test"] for name in ("gaia2", "swe_bench_pro", "terminal_bench_2")
        ) * 2 * paper["test_repetitions"],
        "paper_gaia2_optimizer_cost_usd": str(
            Decimal("14.56") * protocol["paper_cost_evidence"]["generated_patches"]
        ),
        "models": [],
        "errors": [],
    }
    if args.online:
        try:
            catalog = {m["id"]: m for m in get_metadata("models")["data"]}
            for model_id in MODELS:
                model = catalog.get(model_id)
                if model is None:
                    report["models"].append({"id": model_id, "available": False})
                    continue
                pricing = model["pricing"]
                report["models"].append({
                    "id": model_id,
                    "available": True,
                    "context_length": model.get("context_length"),
                    "tools_supported": "tools" in model.get("supported_parameters", []),
                    "pricing_per_token_usd": pricing,
                    "gaia2_paper_token_volume_cost_no_cache_usd": str(
                        Decimal(pricing["prompt"]) * 550988
                        + Decimal(pricing["completion"]) * 7380
                    ),
                    "estimate_note": "Illustration only: token counts change with model; excludes judge and retries, cache discounts, request fees and other charges",
                })
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError):
            report["errors"].append("Public model metadata unavailable")
        if key:
            try:
                data = get_metadata("key", key)["data"]
                report["account"] = {name: data.get(name) for name in (
                    "limit", "limit_remaining", "limit_reset", "usage", "is_free_tier"
                )}
                report["account"]["note"] = "Key limit is not necessarily account credit balance"
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError):
                report["errors"].append("Current-key metadata unavailable")
        else:
            report["errors"].append("OpenRouter key missing")
    result = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
