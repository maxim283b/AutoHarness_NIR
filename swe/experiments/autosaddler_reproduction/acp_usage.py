"""Pure ACP turn-usage normalization, usable in both experiment Python envs."""


def normalize_acp(result):
    raw = result.get("usage") or {}
    inputs, outputs = raw.get("inputTokens"), raw.get("outputTokens")
    complete = all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (inputs, outputs))
    if not complete:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None, "usage_complete": False, "raw": raw}
    read, write, reasoning = raw.get("cachedReadTokens", 0), raw.get("cachedWriteTokens", 0), raw.get("thoughtTokens", 0)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (read, write, reasoning)) or reasoning > outputs:
        raise ValueError("Invalid ACP cache/reasoning counters")
    input_total = inputs + read + write
    total = input_total + outputs
    if raw.get("totalTokens", total) != total:
        raise ValueError("Inconsistent ACP turn counters")
    return {"input_tokens": input_total, "uncached_input_tokens": inputs, "output_tokens": outputs, "total_tokens": total,
        "cached_input_tokens": read, "cache_write_tokens": write, "reasoning_tokens": reasoning, "usage_complete": True, "raw": raw}
