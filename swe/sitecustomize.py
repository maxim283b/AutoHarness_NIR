"""Load the operator's validation image in future isolated optimizer workers.

The active experiment worker can keep running; each optimizer child is a fresh
Python process and imports this file through its fixed PYTHONPATH.
"""
import json
import os
from pathlib import Path

_journal = os.environ.get("AUTOHARNESS_GAIA_RESOURCE_JOURNAL", "")
if Path(_journal).name.startswith("optimizer_owned_resources-"):
    from improved_v2.optimizer_child_overlay import BASE_IMAGE_ID, IMAGE_ID
    from experiments.autosaddler_reproduction import gaia_owned_resources, optimizer_provider

    optimizer_provider.IMAGE = IMAGE_ID
    gaia_owned_resources.NATIVE_IMAGE_ID = IMAGE_ID
    Path(_journal).parent.joinpath("operator-image.json").write_text(json.dumps({
        "base_image_id": BASE_IMAGE_ID,
        "optimizer_image_id": IMAGE_ID,
        "change": "Python jsonschema 4.26.0 and dependencies only",
        "activation": "isolated-child-startup",
    }, sort_keys=True) + "\n")
