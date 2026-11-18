"""Run model code against MBPP asserts in a separate Python process.

This is the evaluator. It is deterministic given the code: each assert is executed
independently after the solution is exec'd, inside a fresh subprocess with a timeout.
No `signal`/`resource` (not available on Windows); isolation = subprocess + temp cwd + timeout.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_TIMEOUT_S = 10.0
_MARKER = "__SANDBOX_RESULT__"

_RUNNER = r'''
import json, sys, traceback
payload = json.load(open(sys.argv[1], encoding="utf-8"))
res = {"setup_error": None, "asserts": []}
ns = {"__name__": "__main__"}
try:
    exec(compile("\n".join(payload["imports"]) + "\n" + payload["code"], "solution.py", "exec"), ns)
except BaseException as e:
    res["setup_error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
if res["setup_error"] is None:
    for a in payload["asserts"]:
        try:
            exec(compile(a, "test.py", "exec"), ns)
            res["asserts"].append(None)
        except BaseException as e:
            msg = "".join(traceback.format_exception_only(type(e), e)).strip()
            res["asserts"].append(msg or type(e).__name__)
sys.stdout.flush()
print("\n''' + _MARKER + r'''" + json.dumps(res))
'''


@dataclass
class SandboxResult:
    ok: bool
    n_passed: int
    n_total: int
    error: str | None  # first error (setup error, first failing assert, timeout) or None
    timed_out: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def run_code_with_asserts(
    code: str, imports: list[str], asserts: list[str], timeout: float = DEFAULT_TIMEOUT_S
) -> SandboxResult:
    n = len(asserts)
    with tempfile.TemporaryDirectory(prefix="mbpp_sbx_") as td:
        tdp = Path(td)
        (tdp / "runner.py").write_text(_RUNNER, encoding="utf-8")
        (tdp / "payload.json").write_text(
            json.dumps({"code": code, "imports": imports, "asserts": asserts}), encoding="utf-8"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "runner.py", "payload.json"],
                cwd=td,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(False, 0, n, f"Timeout: exceeded {timeout:.0f}s", timed_out=True)

    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith(_MARKER)), None)
    if line is None:
        err = (proc.stderr or "").strip().splitlines()
        return SandboxResult(False, 0, n, f"Process exited ({proc.returncode}) without result: " + (err[-1] if err else ""))
    res = json.loads(line[len(_MARKER):])
    if res["setup_error"] is not None:
        return SandboxResult(False, 0, n, res["setup_error"])
    errs = res["asserts"]
    n_passed = sum(e is None for e in errs)
    first_err = next((e for e in errs if e is not None), None)
    return SandboxResult(n_passed == n, n_passed, n, first_err)
