"""Create a byte-for-byte, content-addressed archive of the successful official run.

The source run is never modified.  The archive is created only when the destination
does not exist, and every copied file is checked against its source SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/final_harness_experiment/autosaddler_official_smoke/codex-official-smoke-v5"
DEST = ROOT / "results/final_harness_experiment/official_baseline_20261012"
ARTIFACT = DEST / "artifact"
CONFIG = ROOT / "experiments/final_harness_experiment/autosaddler_codex_smoke.yaml"
PREREG = ROOT / "experiments/final_harness_experiment/preregistered_protocol.json"
AUTOSADDLER = ROOT / "third_party/AutoSaddler"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files(root: Path) -> list[Path]:
    return sorted((path for path in root.rglob("*") if path.is_file()), key=lambda p: p.as_posix())


def command(*args: str) -> str:
    return subprocess.check_output(
        args, cwd=ROOT, text=True, encoding="utf-8", errors="replace", stderr=subprocess.STDOUT
    ).strip()


def main() -> None:
    if not SOURCE.is_dir():
        raise SystemExit(f"source run not found: {SOURCE}")
    manifest_path = DEST / "archive_manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite immutable archive: {DEST}")
    if ARTIFACT.exists():
        # Safe recovery from an interrupted first creation: verify the existing
        # copy below rather than deleting or overwriting it.
        pass
    else:
        DEST.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SOURCE, ARTIFACT, copy_function=shutil.copy2)
    archived_config = DEST / CONFIG.name
    if not archived_config.exists():
        shutil.copy2(CONFIG, archived_config)

    source_hashes = {p.relative_to(SOURCE).as_posix(): sha256(p) for p in files(SOURCE)}
    archive_hashes = {p.relative_to(ARTIFACT).as_posix(): sha256(p) for p in files(ARTIFACT)}
    if source_hashes != archive_hashes:
        raise RuntimeError("archive verification failed: copied tree differs from source")

    events = [json.loads(line) for line in (SOURCE / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    usages = [event["payload"] for event in events if event["event_type"] == "ModelUsageObserved"]
    completed = [event for event in events if event["event_type"] == "RunCompleted"]
    sessions = []
    for event in events:
        if event["event_type"] != "SessionStarted":
            continue
        payload = event["payload"]
        sessions.append(
            {
                "session_id": payload["session_id"],
                "session_type": payload["logical_operation_id"].split(":")[-1],
                "stage": payload["stage"],
                "request_uri": payload["request"]["uri"],
            }
        )

    usage_by_session = {usage["session_id"]: usage["usage"] for usage in usages}
    for session in sessions:
        usage = usage_by_session[session["session_id"]]
        session["input_tokens"] = usage["input_tokens"]
        session["output_tokens"] = usage["output_tokens"]
        session["model"] = usage["model"]

    candidate_id = completed[0]["payload"]["selected_candidate_id"]
    candidate_file = SOURCE / "candidates" / candidate_id.removeprefix("sha256:") / "candidate.json"
    summary = {
        "schema_version": "official-autosaddler-baseline-archive/v1",
        "classification": "official_infrastructure_smoke_not_scientific_target_validation",
        "immutable_source_run": SOURCE.relative_to(ROOT).as_posix(),
        "run_id": "codex-official-smoke-v5",
        "framework": {
            "name": "AutoSaddler V2",
            "git_commit": command("git", "-C", str(AUTOSADDLER), "rev-parse", "HEAD"),
        },
        "provider": {
            "type": "CodexAgentProvider",
            "codex_cli_version": command("codex", "--version").splitlines()[-1],
            "model": usages[0]["usage"]["model"],
        },
        "scenario": "fake",
        "configuration": {
            "original_path": CONFIG.relative_to(ROOT).as_posix(),
            "archived_path": CONFIG.name,
            "sha256": sha256(CONFIG),
        },
        "original_preregistration": {
            "path": PREREG.relative_to(ROOT).as_posix(),
            "sha256": sha256(PREREG),
        },
        "selected_candidate_id": candidate_id,
        "candidate_file_sha256": sha256(candidate_file),
        "patch": json.loads(candidate_file.read_text(encoding="utf-8")),
        "development": {"seed_score": 0.0, "selected_score": 1.0},
        "run_completed_event": completed[0],
        "proposer_calls": len(usages),
        "proposer_input_tokens": sum(item["usage"]["input_tokens"] for item in usages),
        "proposer_output_tokens": sum(item["usage"]["output_tokens"] for item in usages),
        "sessions": sessions,
        "tree_file_count": len(source_hashes),
        "tree_files": source_hashes,
        "verification": "all archived artifact files matched source SHA-256 at archive creation",
    }
    manifest_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("run_id", "selected_candidate_id", "proposer_calls", "proposer_input_tokens", "proposer_output_tokens", "tree_file_count")}, indent=2))


if __name__ == "__main__":
    main()
