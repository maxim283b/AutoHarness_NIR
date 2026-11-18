from sandbox import run_code_with_asserts as run


def test_ok():
    assert run("def f(x): return x + 1", [], ["assert f(1) == 2", "assert f(2) == 3"])[:2] == (True, 2)


def test_syntax_error():
    ok, n, err, _ = run("def f(:", [], ["assert f(1) == 2"])
    assert not ok and n == 0 and "SyntaxError" in err


def test_timeout():
    ok, n, err, _ = run("while True: pass", [], ["assert True"], timeout=2)
    assert not ok and "Timeout" in err


def test_partial_and_actual_value():
    ok, n, err, per = run("def f(x): return x", [], ["assert f(1) == 1", "assert f(2) == 3"])
    assert not ok and n == 1 and "got 2" in err


def test_imports():
    assert run("def f(x): return math.sqrt(x)", ["import math"], ["assert f(4) == 2"])[0]
