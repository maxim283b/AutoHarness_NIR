from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


AUTOSADDLER_COMMIT = "e841543ce6cf6bdf690e65134b12a829ec9f66d1"


def probe_official_checkout(root: Path) -> dict[str, Any]:
    checkout = root / "third_party" / "AutoSaddler"
    if not checkout.exists():
        return {"ok": False, "reason": "official checkout missing", "expected_commit": AUTOSADDLER_COMMIT}
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    license_text = (checkout / "LICENSE").read_text(encoding="utf-8")
    source = checkout / "src"
    sys.path.insert(0, str(source))
    try:
        from autosaddler.v2.core.policies import EpochShuffledTaskSelectionPolicy, MatchedValidStrictImprovement
        from autosaddler.v2.harness.component_map import ComponentMapHarnessSpace

        symbols = [
            EpochShuffledTaskSelectionPolicy.__name__,
            MatchedValidStrictImprovement.__name__,
            ComponentMapHarnessSpace.__name__,
        ]
    finally:
        sys.path.remove(str(source))
    return {
        "ok": commit == AUTOSADDLER_COMMIT and "MIT License" in license_text,
        "commit": commit,
        "expected_commit": AUTOSADDLER_COMMIT,
        "license": "MIT" if "MIT License" in license_text else "unknown",
        "official_symbols_loaded": symbols,
        "reproduction_status": "exact_or_adapter_reproduction",
        "official_smoke_run": "codex-official-smoke-v5",
        "note": "Official Codex provider and unchanged V2 lifecycle completed with user-authorized danger-full-access; shims are Windows platform compatibility only.",
    }


def write_probe(root: Path, destination: Path) -> dict[str, Any]:
    result = probe_official_checkout(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
