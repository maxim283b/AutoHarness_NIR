"""Run model code against asserts in a separate Python process (Windows-safe: timeout only, no signals)."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap

_RUNNER = textwrap.dedent(
    """
    import ast, json, sys, traceback
    _asserts = json.loads(sys.argv[1])

    def _explain(a, g):
        # For `assert LHS == RHS`, report the actual LHS value.
        try:
            t = ast.parse(a).body[0].test
            if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq):
                got = eval(compile(ast.Expression(t.left), "<a>", "eval"), g)
                return "AssertionError: expected %s, got %s" % (ast.unparse(t.comparators[0])[:200], repr(got)[:200])
        except BaseException:
            pass
        return "AssertionError"

    _res = {"load_error": None, "results": []}
    _g = {"__name__": "__sandbox__"}
    try:
        exec(compile(open(sys.argv[2], encoding="utf-8").read(), "solution.py", "exec"), _g)
    except BaseException as e:
        _res["load_error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
    if _res["load_error"] is None:
        for _a in _asserts:
            try:
                exec(_a, _g)
                _res["results"].append([True, None])
            except AssertionError:
                _res["results"].append([False, _explain(_a, _g)])
            except BaseException as e:
                _msg = "".join(traceback.format_exception_only(type(e), e)).strip()
                _res["results"].append([False, _msg or type(e).__name__])
    sys.stdout.write("\\n@@RESULT@@" + json.dumps(_res))
    """
)


def run_code_with_asserts(
    code: str, imports: list[str], asserts: list[str], timeout: float = 10.0
) -> tuple[bool, int, str | None, list[tuple[bool, str | None]]]:
    """Returns (all_ok, n_passed, first_error, per_assert)."""
    src = "\n".join(imports) + "\n" + (code or "")
    with tempfile.TemporaryDirectory() as d:
        sol = os.path.join(d, "solution.py")
        runner = os.path.join(d, "runner.py")
        with open(sol, "w", encoding="utf-8") as f:
            f.write(src)
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_RUNNER)
        try:
            p = subprocess.run(
                [sys.executable, "-I", runner, json.dumps(asserts), sol],
                cwd=d,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            err = f"Timeout: execution exceeded {timeout:.0f}s (possible infinite loop)"
            return False, 0, err, [(False, err)] * len(asserts)
    if "@@RESULT@@" not in p.stdout:
        err = (p.stderr or "process crashed").strip().splitlines()[-1:] or ["process crashed"]
        return False, 0, err[0], [(False, err[0])] * len(asserts)
    res = json.loads(p.stdout.rsplit("@@RESULT@@", 1)[1])
    if res["load_error"]:
        return False, 0, res["load_error"], [(False, res["load_error"])] * len(asserts)
    per = [(ok, e) for ok, e in res["results"]]
    n = sum(ok for ok, _ in per)
    first_err = next((f"{a}\n-> {e}" for a, (ok, e) in zip(asserts, per) if not ok), None)
    return n == len(asserts), n, first_err, per
