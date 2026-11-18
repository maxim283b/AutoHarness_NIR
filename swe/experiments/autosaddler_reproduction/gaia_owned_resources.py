"""Journal and clean only GAIA native containers bearing an exact owner label."""
import json
import hashlib
import os
from pathlib import Path
import re
import subprocess
import threading
import uuid

from .gaia_native_assets import NATIVE_IMAGE_ID

OWNER_ENV = "AUTOHARNESS_GAIA_WORKER_OWNER"
JOURNAL_ENV = "AUTOHARNESS_GAIA_RESOURCE_JOURNAL"
OWNER_LABEL = "autoharness.gaia-owner"
KIND_LABEL = "autoharness.gaia-kind"
KIND = "native-target"
KINDS = {"native-target": r"autoharness-target-[0-9a-f]{12}",
         "native-optimizer": r"autoharness-optimizer-[0-9a-f]{12}",
         "source-build": r"autoharness-source-build-[0-9a-f]{12}"}

def validate_resource(name, kind):
    if kind not in KINDS or re.fullmatch(KINDS[kind], name) is None:
        raise ValueError("Invalid GAIA native resource kind/name")
_lock = threading.Lock()


def validate_owner(owner):
    if not isinstance(owner, str) or re.fullmatch(r"[0-9a-f]{32}", owner) is None:
        raise ValueError("Invalid GAIA resource owner identity")


def _append(path, value):
    with _lock, Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(value) + "\n")
        file.flush()
        os.fsync(file.fileno())


def register_container(name, artifacts, kind=KIND):
    validate_resource(name, kind)
    owner = os.environ.get(OWNER_ENV) or uuid.uuid4().hex
    validate_owner(owner)
    journal = Path(os.environ.get(JOURNAL_ENV) or (Path(artifacts) / "owned_resources.jsonl")).resolve()
    if not journal.parent.is_dir():
        raise ValueError("GAIA resource journal parent is missing")
    record = {"kind": kind, "name": name, "owner": owner,
              "image": NATIVE_IMAGE_ID, "phase": "intent"}
    _append(journal, record)  # Before Docker starts; survives worker cancellation.
    return {**record, "journal": str(journal)}


def container_labels(record):
    validate_owner(record["owner"])
    validate_resource(record["name"], record.get("kind", KIND))
    return ["--label", OWNER_LABEL + "=" + record["owner"],
            "--label", KIND_LABEL + "=" + record.get("kind", KIND)]


def process_creation_identity(pid):
    """Windows kernel creation ticks prevent killing a reused PID."""
    if os.name != "nt":
        import psutil
        try:
            process = psutil.Process(pid)
            if process.status() == psutil.STATUS_ZOMBIE:
                return None
            return process.create_time()
        except psutil.NoSuchProcess:
            return None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:  # No process object for this PID.
            return None
        raise OSError("Cannot verify owned GAIA process identity")
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            raise OSError("Cannot read owned GAIA process creation time")
        # A terminated process object may remain while its parent holds a handle.
        exited = (times[1].dwHighDateTime << 32) | times[1].dwLowDateTime
        if exited:
            return None
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


def register_native_process(process, record):
    identity = process_creation_identity(process.pid)
    if identity is None:
        return
    _append(record["journal"], {"kind": "native-client", "owner": record["owner"],
        "pid": process.pid, "creation_identity": identity, "phase": "started"})


def cleanup_native_process(row):
    pid = row["pid"]
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError("Invalid owned process PID")
    current = process_creation_identity(pid)
    if current is None or current != row["creation_identity"]:
        return {"pid": pid, "original_process_absent": True, "reused_pid_untouched": current is not None}
    if os.name != 'nt':
        import psutil
        try:
            process = psutil.Process(pid)
            if process.create_time() != row['creation_identity']:
                return {'pid': pid, 'original_process_absent': True, 'reused_pid_untouched': True}
            children = process.children(recursive=True)
            for child in children:
                try: child.kill()
                except psutil.NoSuchProcess: pass
            process.kill()
            psutil.wait_procs(children + [process], timeout=10)
        except psutil.NoSuchProcess:
            pass
        if process_creation_identity(pid) == row['creation_identity']:
            raise RuntimeError('Owned native process remains alive')
        return {'pid': pid, 'original_process_absent': True}
    result = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=20)
    after = process_creation_identity(pid)
    if after == row["creation_identity"]:
        raise RuntimeError("Owned GAIA process did not terminate")
    return {"pid": pid, "original_process_absent": True, "taskkill_returncode": result.returncode}


def _inspect(name):
    # Read only identity and our two labels; never collect container ENV/mounts.
    identity = ('{"Name":{{json .Name}},"Image":{{json .Image}},"Config":{"Labels":{'
                '"autoharness.gaia-owner":{{json (index .Config.Labels "autoharness.gaia-owner")}},'
                '"autoharness.gaia-kind":{{json (index .Config.Labels "autoharness.gaia-kind")}}}}}')
    result = subprocess.run(["docker", "container", "inspect", name, "--format", identity],
                            capture_output=True, timeout=15)
    if result.returncode:
        stderr = result.stderr.lower()
        if b"no such" in stderr or b"not found" in stderr:
            return None
        raise RuntimeError("Could not verify GAIA container identity")
    return json.loads(result.stdout)


def cleanup_container(record):
    owner, name = record["owner"], record["name"]
    validate_owner(owner)
    kind = record.get("kind", KIND)
    validate_resource(name, kind)
    item = _inspect(name)
    if item is not None:
        labels = item.get("Config", {}).get("Labels") or {}
        if (item.get("Name", "").lstrip("/") != name or labels.get(OWNER_LABEL) != owner
                or labels.get(KIND_LABEL) != kind or item.get("Image") != NATIVE_IMAGE_ID):
            raise RuntimeError("Refusing container with mismatched GAIA ownership/image")
        removed = subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
        absent = _inspect(name) is None
        if record.get("journal"):
            _append(record["journal"], {"kind": kind, "name": name, "owner": owner,
                "phase": "removal_checked", "remove_returncode": removed.returncode,
                "stderr_bytes": len(removed.stderr or b""),
                "stderr_sha256": hashlib.sha256(removed.stderr or b"").hexdigest(),
                "confirmed_absent": absent})
        # Docker --rm may remove an already identity-verified container between
        # inspect and rm. Only a fresh explicit absence proves cleanup complete.
        if not absent:
            raise RuntimeError("Owned GAIA container cleanup did not complete")
    if record.get("journal"):
        _append(record["journal"], {"kind": kind, "name": name, "owner": owner, "phase": "confirmed_absent"})
    return {"name": name, "confirmed_absent": True}


def cleanup_owner(owner, journal):
    validate_owner(owner)
    names = set()
    processes = {}
    incomplete_journal_lines = 0
    journal = Path(journal)
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                incomplete_journal_lines += 1
                continue  # Label discovery still covers a torn final intent line.
            if row.get("owner") != owner or row.get("kind") not in set(KINDS) | {"native-client"}:
                raise ValueError("Resource journal ownership mismatch")
            if row["kind"] in KINDS:
                validate_resource(row["name"], row["kind"])
                names.add((row["name"], row["kind"]))
            else:
                processes[(row["pid"], row["creation_identity"])] = row
    stopped = [cleanup_native_process(row) for row in processes.values()]
    for kind in KINDS:
        result = subprocess.run(["docker", "ps", "-a", "--filter", "label=" + OWNER_LABEL + "=" + owner,
                                 "--filter", "label=" + KIND_LABEL + "=" + kind, "--format", "{{.Names}}"],
                                capture_output=True, timeout=15)
        if result.returncode:
            raise RuntimeError("Could not enumerate owned GAIA resources")
        names.update((name, kind) for name in result.stdout.decode("utf-8").splitlines())
    cleaned = [cleanup_container({"owner": owner, "name": name, "kind": kind, "journal": str(journal)})
               for name, kind in sorted(names)]
    return {"owned_containers": cleaned, "native_processes": stopped, "incomplete_journal_lines": incomplete_journal_lines,
            "confirmed_absent": True}
