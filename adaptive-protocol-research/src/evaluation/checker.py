from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRATCH = Path(__file__).resolve().parents[2] / ".scratch"

ALLOWED_IMPORTS = {"math", "collections", "itertools", "functools", "re", "heapq", "bisect", "operator", "string", "statistics", "array", "copy"}
BLOCKED_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals"}


def check_source(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in ALLOWED_IMPORTS for alias in node.names):
                return "disallowed import"
        if isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] not in ALLOWED_IMPORTS:
                return "disallowed import"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            return "disallowed call"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "disallowed dunder attribute"
    return None


RUNNER = '''import json, sys, traceback
ns = {"__name__": "__main__"}
try:
    exec(compile(open(sys.argv[1], encoding="utf8").read(), "solution", "exec"), ns)
    for test in json.loads(sys.argv[2]):
        exec(test, ns)
    print("@@RESULT@@" + json.dumps({"ok": True, "error": ""}))
except BaseException as exc:
    print("@@RESULT@@" + json.dumps({"ok": False, "error": "".join(traceback.format_exception_only(type(exc), exc)).strip()[:500]}))
'''


def run_tests(code: str, imports: list[str], tests: list[str], timeout: float = 3.0) -> tuple[bool, str]:
    problem = check_source(code)
    if problem:
        return False, problem
    SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
        path = Path(temp)
        solution = path / "solution.py"
        runner = path / "runner.py"
        solution.write_text("\n".join(imports) + "\n" + code, encoding="utf8")
        runner.write_text(RUNNER, encoding="utf8")
        try:
            result = subprocess.run([sys.executable, "-I", str(runner), str(solution), json.dumps(tests)], cwd=path, capture_output=True, text=True, timeout=timeout, encoding="utf8", errors="replace")
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if "@@RESULT@@" not in result.stdout:
        return False, "process failed: " + result.stderr[-300:]
    data = json.loads(result.stdout.rsplit("@@RESULT@@", 1)[1])
    return bool(data["ok"]), str(data["error"])
