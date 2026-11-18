"""Explicit benchmark scope, shared by search, validation and final evaluation."""
BENCHMARKS = ("gaia2", "swe_bench_pro", "terminal_bench_2")


def selected_benchmarks(settings):
    values = settings.get("benchmarks", BENCHMARKS)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("benchmarks must be a nonempty list")
    if any(not isinstance(value, str) or value not in BENCHMARKS for value in values):
        raise ValueError("Unknown benchmark selection")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate benchmark selection")
    return tuple(values)
