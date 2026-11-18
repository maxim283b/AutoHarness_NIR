"""Operator-only accounting support for Go Plus Extra Usage.

Loaded only after the owner confirms credits and Extra Usage are active.
The frozen experiment and model traffic remain unchanged.
"""
from __future__ import annotations

import math

from experiments.autosaddler_reproduction import go_gateway

ORIGINAL_METERED = go_gateway.metered_usage


def quota_with_extra_usage(client, auth, ceiling=100):
    response = client.get("https://opencode.ai/zen/go/v1/usage", headers={
        "Authorization": "Bearer " + auth["access"],
        "x-opencode-org-id": auth["metadata"]["orgID"],
        "User-Agent": "autoharness-opencode/1.19.5",
    }, timeout=25)
    if response.status_code != 200:
        raise RuntimeError("Go quota endpoint returned HTTP " + str(response.status_code))
    value = response.json()["usage"]
    for window in ("rolling", "weekly", "monthly"):
        row = value[window]
        percent = row.get("percent")
        if (row.get("status") not in {"ok", "rate-limited"}
                or isinstance(percent, bool)
                or not isinstance(percent, (int, float))
                or not math.isfinite(percent) or percent < 0):
            raise RuntimeError("Invalid Go quota metadata: " + window)
    return value


def metered_usage_with_balance(events):
    merged = {}
    for event in events:
        for value in (event.get("usage"), (event.get("message") or {}).get("usage")):
            if isinstance(value, dict):
                merged.update(value)
    billing = merged.get("opencode") or {}
    source = billing.get("billing_source")
    if source == "go-plus":
        return ORIGINAL_METERED(events)
    if source not in {"go", "balance"}:
        raise RuntimeError("Unexpected billing source for frozen Go models")
    cost = merged.get("cost")
    list_cost = billing.get("list_cost")
    if (isinstance(cost, bool) or not isinstance(cost, (float, int))
            or not math.isfinite(cost) or cost < 0
            or isinstance(list_cost, bool) or not isinstance(list_cost, (float, int))
            or not math.isfinite(list_cost) or list_cost < 0):
        raise RuntimeError("Paid response omitted valid USD cost")
    tokens = billing.get("tokens")
    if not isinstance(tokens, dict) or "input" not in tokens or "output" not in tokens:
        raise RuntimeError("Paid response omitted billed token counters")
    for name in ("input", "output", "reasoning", "cache_read", "cache_write_5m", "cache_write_1h"):
        value = tokens.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError("Invalid paid token counter")
    allowance = billing.get("allowance") if source == "go" else 0
    if source == "go" and (cost != 0 or isinstance(allowance, bool)
                            or not isinstance(allowance, (int, float))
                            or not math.isfinite(allowance) or allowance < 0):
        raise RuntimeError("Go response omitted valid allowance or charged balance")
    return {"allowance": allowance, "billing_source": source,
            "balance_cost": cost if source == "balance" else 0,
            "list_cost": list_cost, "tokens": tokens,
            "total_tokens": sum(tokens.get(k, 0) for k in
                                ("input", "output", "cache_read", "cache_write_5m", "cache_write_1h"))}


def install():
    go_gateway.quota = quota_with_extra_usage
    go_gateway.metered_usage = metered_usage_with_balance
