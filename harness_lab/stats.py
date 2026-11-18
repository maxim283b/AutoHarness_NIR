"""Paired statistics for small benchmark comparisons."""
from __future__ import annotations

import math
import random
from collections.abc import Sequence


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p-value; b = A-only wins, c = B-only wins."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_compare(a: Sequence[float], b: Sequence[float], *, boots: int = 5000, seed: int = 0) -> dict:
    """B - A on aligned per-task scores in {0,1}."""
    assert len(a) == len(b) and a
    n = len(a)
    diffs = [y - x for x, y in zip(a, b)]
    only_a = sum(1 for x, y in zip(a, b) if x > y)
    only_b = sum(1 for x, y in zip(a, b) if y > x)
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(boots))
    return {
        "n": n,
        "mean_a": sum(a) / n,
        "mean_b": sum(b) / n,
        "delta": sum(diffs) / n,
        "ci95": [means[int(0.025 * boots)], means[int(0.975 * boots) - 1]],
        "a_only": only_a,
        "b_only": only_b,
        "p_mcnemar": mcnemar_exact(only_a, only_b),
    }
