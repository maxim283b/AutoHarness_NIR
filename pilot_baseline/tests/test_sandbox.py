from sandbox import run_code_with_asserts


def test_success():
    r = run_code_with_asserts("def add(a, b):\n    return a + b", [], ["assert add(1, 2) == 3", "assert add(0, 0) == 0"])
    assert r.ok and r.n_passed == 2 and r.error is None


def test_partial_failure_counts():
    r = run_code_with_asserts("def f(x):\n    return x", [], ["assert f(1) == 1", "assert f(2) == 3"])
    assert not r.ok and r.n_passed == 1 and "AssertionError" in r.error


def test_syntax_error():
    r = run_code_with_asserts("def f(:\n  pass", [], ["assert f() is None"])
    assert not r.ok and r.n_passed == 0 and "SyntaxError" in r.error


def test_infinite_loop_times_out():
    r = run_code_with_asserts("def f():\n    while True: pass", [], ["assert f()"], timeout=2)
    assert r.timed_out and not r.ok


def test_imports_and_sys_exit():
    r = run_code_with_asserts("def f():\n    return math.pi", ["import math"], ["assert abs(f() - 3.14159) < 1e-3"])
    assert r.ok
    r = run_code_with_asserts("import sys\nsys.exit(0)", [], ["assert True"])
    assert not r.ok and "SystemExit" in r.error
