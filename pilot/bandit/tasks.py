"""Task pools for the bandit pilot, built deterministically from the cached MBPP-sanitized file.

  online  (100): MBPP 'validation' (43) + first 57 of the seed-0-shuffled MBPP 'train'.
                 Contains the pilot's train(30) and val(20) splits. The bandit learns ONLY here.
  heldout  (50): exactly the pilot's frozen test split (same ids as runs/baseline), never used for updates.
  memory   (70): remaining 63 MBPP 'train' tasks + 7 'prompt' tasks with reference code; disjoint from both.
"""

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_raw, load_splits  # noqa: E402

N_ONLINE_TRAIN = 57

# Coarse task types used only for analysis and for the contextual-lite per-type policy.
_TYPE_RULES = [
    ("string", r"\b(string|strings|character|characters|char|word|words|vowel|vowels|substring|uppercase|lowercase|"
               r"letter|letters|sentence|regex|palindrome|text)\b"),
    ("math", r"\b(number|numbers|integer|integers|digit|digits|prime|factorial|sum|area|volume|perimeter|nth|"
             r"divisor|divisors|binary|square|squares|even|odd|bit|bits|power|sqrt|average|product|fibonacci|"
             r"triangle|circle|cube|decimal|gcd|lcm)\b"),
    ("sequence", r"\b(list|lists|array|arrays|tuple|tuples|sublist|element|elements|sort|sorted|dictionary|dict|"
                 r"set|matrix|pairs?|items?)\b"),
]


def task_type(prompt: str) -> str:
    p = prompt.lower()
    hits = [name for name, rx in _TYPE_RULES if re.search(rx, p)]
    if not hits:
        return "other"
    # strings and sequences are the structural object; math words appear everywhere ("sum of the list")
    if "string" in hits:
        return "string"
    if "sequence" in hits:
        return "sequence"
    return hits[0]


def build_pools() -> dict[str, list[dict]]:
    raw = load_raw()
    splits = load_splits()  # verifies the frozen manifest / data hash
    train = sorted(raw["train"], key=lambda r: r["task_id"])
    random.Random(0).shuffle(train)  # same permutation as data.make_splits -> pilot train ⊂ online
    online = sorted(raw["validation"], key=lambda r: r["task_id"]) + train[:N_ONLINE_TRAIN]
    memory = train[N_ONLINE_TRAIN:] + raw["prompt"]
    heldout = splits["test"]
    ids = [{t["task_id"] for t in p} for p in (online, heldout, memory)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2]), "pools overlap"
    assert {t["task_id"] for t in splits["train"] + splits["val"]} <= ids[0]
    for p in (online, heldout):
        for t in p:
            t["type"] = task_type(t["prompt"])
    return {"online": online, "heldout": heldout, "memory": memory}
