"""Operator-only optimizer image overlay with Python jsonschema.

The frozen optimizer implementation and target image are unchanged. The
derived image adds only Python validation dependencies to optimizer sessions.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

BASE_IMAGE_ID = "sha256:c0e9d26f1c27e4a9ad7db77eaafa1f836af5cdbcaf0e6d13382f31c03818e7e5"
IMAGE_ID = "sha256:1bed6b7004e0b10b96fbf0d14d74d5e5bd90db84fffc93c05e6580421593e267"
MODULE = "experiments.autosaddler_reproduction.gaia_optimizer_supervisor"


def install() -> None:
    from experiments.autosaddler_reproduction import gaia_optimizer_supervisor as supervisor
    from experiments.autosaddler_reproduction import gaia_owned_resources

    image = subprocess.run(["docker", "image", "inspect", IMAGE_ID, "--format", "{{.Id}}"],
                           capture_output=True, text=True, timeout=20, check=True)
    if image.stdout.strip() != IMAGE_ID:
        raise RuntimeError("Operator optimizer image identity mismatch")
    gaia_owned_resources.NATIVE_IMAGE_ID = IMAGE_ID
    original_invoke = supervisor.invoke

    def invoke_with_overlay(command, env, result_path, artifacts, timeout,
                            cancellation=None, expected_model=None):
        if len(command) < 3 or command[1:3] != ["-m", MODULE]:
            raise RuntimeError("Unexpected optimizer child launch command")
        Path(artifacts, "operator-image.json").write_text(json.dumps({
            "base_image_id": BASE_IMAGE_ID, "optimizer_image_id": IMAGE_ID,
            "change": "Python jsonschema 4.26.0 and dependencies only",
        }, sort_keys=True) + "\n")
        patched = [command[0], "-m", "improved_v2.optimizer_child_overlay", *command[3:]]
        return original_invoke(patched, env, result_path, artifacts, timeout,
                               cancellation, expected_model)

    supervisor.invoke = invoke_with_overlay


def main() -> None:
    from experiments.autosaddler_reproduction import gaia_owned_resources, optimizer_provider
    from experiments.autosaddler_reproduction.gaia_optimizer_supervisor import child_main

    optimizer_provider.IMAGE = IMAGE_ID
    gaia_owned_resources.NATIVE_IMAGE_ID = IMAGE_ID
    child_main()


if __name__ == "__main__":
    main()
