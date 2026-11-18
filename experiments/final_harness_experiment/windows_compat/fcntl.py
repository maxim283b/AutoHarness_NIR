"""Import-only Windows shim for AutoSaddler's unused Meta-ARE provisioning path.

The official V2 registry imports Meta-ARE provisioning even when the selected
scenario is `fake`.  That module imports POSIX-only ``fcntl``.  The official
Codex smoke never calls flock; fail loudly if a future path does.
"""

LOCK_EX = 2
LOCK_UN = 8


def flock(*_args, **_kwargs):
    raise RuntimeError("POSIX flock was invoked through the Windows import-only shim")
