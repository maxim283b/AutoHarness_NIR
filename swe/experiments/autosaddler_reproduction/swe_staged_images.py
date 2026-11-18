"""Prepare one digest-pinned SWE image before its first model attempt."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from .swe_adapter import ROOT


PROOFS = ROOT / "results/autosaddler_reproduction/swe_staged_images"
MIN_FREE_BYTES = 24 * 1024**3


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _checked(container, command, **kwargs) -> str:
    outcome = container.exec_run(command, **kwargs)
    if outcome.exit_code:
        raise RuntimeError("SWE image setup failed: " + outcome.output.decode(errors="replace")[-1000:])
    return outcome.output.decode(errors="replace").strip()


def ensure_image(task_id: str, row: dict, digest_file: str | Path, *, allow_pull: bool) -> dict:
    """Return official image metadata after no-inference setup verification.

    The caller may enable a digest-pinned pull. No image deletion occurs here.
    Existing proofs keep the same image ID across all H0/H1 repetitions.
    """
    import docker

    digest_path = (ROOT / digest_file).resolve()
    if not digest_path.is_relative_to(ROOT.resolve()):
        raise RuntimeError("SWE digest plan escapes the isolated runtime")
    raw = digest_path.read_bytes()
    plan = json.loads(raw)
    if plan.get("status") != "all_digests_resolved_no_inference" or plan.get("model_calls") != 0 or plan.get("expected_tasks") != 199:
        raise RuntimeError("All SWE image digests must be resolved before inference")
    if len(plan.get("resolved", {})) != 199:
        raise RuntimeError("Incomplete SWE image digest plan")
    selected = plan["resolved"].get(task_id)
    if not selected or row.get("instance_id") != task_id or selected["base_commit"] != row.get("base_commit"):
        raise RuntimeError("SWE task or base commit differs from frozen image plan")
    tag, digest = selected["tag"], selected["registry_digest"]
    reference = "jefzda/sweap-images@" + digest
    proof_path = PROOFS / (hashlib.sha256(task_id.encode()).hexdigest()[:20] + ".json")
    proof = json.loads(proof_path.read_text(encoding="utf-8")) if proof_path.exists() else None
    if proof and any((proof.get(key) != value for key, value in {
        "task_id": task_id, "tag": tag, "registry_digest": digest,
        "base_commit": row["base_commit"], "digest_plan_sha256": hashlib.sha256(raw).hexdigest(),
        "passed": True, "model_calls": 0,
    }.items())):
        raise RuntimeError("SWE image preflight proof drift")
    client = docker.from_env(timeout=300)
    container = None
    try:
        if proof:
            try:
                image = client.images.get(proof["image_id"])
            except docker.errors.ImageNotFound:
                if not allow_pull:
                    raise RuntimeError("Pinned SWE image is absent and pull is disabled")
                if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
                    raise RuntimeError("SWE image pull blocked by free-space floor")
                image = client.images.pull(reference)
            if image.id != proof["image_id"]:
                raise RuntimeError("Re-pulled SWE image ID differs from prior preflight")
            return proof
        try:
            image = client.images.get(reference)
        except docker.errors.ImageNotFound:
            if not allow_pull:
                raise RuntimeError("SWE image is absent and pull is disabled")
            if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
                raise RuntimeError("SWE image pull blocked by free-space floor")
            image = client.images.pull(reference)
        repo_digests = image.attrs.get("RepoDigests", [])
        if reference not in repo_digests:
            raise RuntimeError("Docker image lacks the frozen registry digest")
        container = client.containers.run(image.id, entrypoint="/bin/sh", command=["-c", "sleep infinity"],
                                          detach=True, mem_limit="8g", pids_limit=512)
        repo_dir = _checked(container, ["git", "rev-parse", "--show-toplevel"])
        if not repo_dir.startswith("/"):
            raise RuntimeError("SWE image has no absolute repository directory")
        _checked(container, ["git", "cat-file", "-e", row["base_commit"] + "^{commit}"], workdir=repo_dir)
        _checked(container, ["git", "reset", "--hard", row["base_commit"]], workdir=repo_dir)
        if _checked(container, ["git", "rev-parse", "HEAD"], workdir=repo_dir) != row["base_commit"]:
            raise RuntimeError("SWE base commit reset mismatch")
        proof = {"task_id": task_id, "tag": tag, "registry_digest": digest,
                 "image_id": image.id, "base_commit": row["base_commit"], "repo_dir": repo_dir,
                 "digest_plan_sha256": hashlib.sha256(raw).hexdigest(), "passed": True,
                 "model_calls": 0, "container_removed_before_inference": True,
                 "prepared_at_utc": datetime.now(timezone.utc).isoformat()}
        return proof
    finally:
        if container is not None:
            container.remove(force=True)
            if proof and proof.get("passed") and not proof_path.exists():
                _save(proof_path, proof)
        client.close()
