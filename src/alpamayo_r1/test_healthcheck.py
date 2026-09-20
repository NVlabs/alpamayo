# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``alpamayo_r1.healthcheck``.

Run:
    pytest src/alpamayo_r1/test_healthcheck.py -v
"""

from __future__ import annotations

from alpamayo_r1.healthcheck import (
    ALL_CHECKS,
    CheckResult,
    _fail,
    _ok,
    _skip,
    _format_line,
    main,
    run,
)


def test_check_result_is_immutable() -> None:
    """CheckResult is a frozen dataclass; mutation raises."""
    r = _ok("foo", "bar")
    try:
        r.detail = "baz"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("CheckResult should be frozen")


def test_format_line_includes_glyph_and_name_and_detail() -> None:
    line = _format_line(_ok("torch", "torch==2.8.0"))
    assert "[ OK ]" in line
    assert "torch" in line
    assert "torch==2.8.0" in line

    line = _format_line(_fail("hf_auth", "401"))
    assert "[FAIL]" in line and "hf_auth" in line and "401" in line

    line = _format_line(_skip("cuda", "no GPU"))
    assert "[SKIP]" in line


def test_all_checks_is_a_non_empty_tuple_of_callables() -> None:
    assert isinstance(ALL_CHECKS, tuple)
    assert len(ALL_CHECKS) >= 5
    for c in ALL_CHECKS:
        assert callable(c)


def test_run_isolates_buggy_checks() -> None:
    """A check that raises should be reported as fail, not crash run()."""
    def boom() -> CheckResult:
        raise RuntimeError("nope")

    def ok() -> CheckResult:
        return _ok("ok", "fine")

    results = run((boom, ok))

    assert len(results) == 2
    assert results[0].status == "fail"
    assert "RuntimeError" in results[0].detail
    assert results[1].status == "pass"


def test_run_preserves_order() -> None:
    def a() -> CheckResult:
        return _ok("a", "first")

    def b() -> CheckResult:
        return _ok("b", "second")

    results = run((a, b))
    assert [r.name for r in results] == ["a", "b"]


def test_main_returns_failure_count() -> None:
    """main()'s return value is the number of failed checks (not skips)."""
    # Monkeypatch the ALL_CHECKS the script reads via module-level rebinding.
    import alpamayo_r1.healthcheck as hc

    def one_pass() -> CheckResult:
        return _ok("one", "ok")

    def one_fail() -> CheckResult:
        return _fail("two", "broken")

    def one_skip() -> CheckResult:
        return _skip("three", "n/a")

    original = hc.ALL_CHECKS
    hc.ALL_CHECKS = (one_pass, one_fail, one_skip)
    try:
        rc = main(argv=[])
    finally:
        hc.ALL_CHECKS = original

    assert rc == 1, f"expected exit code 1 (one fail), got {rc}"


def test_main_quiet_returns_same_failure_count() -> None:
    """--quiet must not affect the exit code, only what's printed."""
    import alpamayo_r1.healthcheck as hc

    def one_fail() -> CheckResult:
        return _fail("x", "broken")

    original = hc.ALL_CHECKS
    hc.ALL_CHECKS = (one_fail,)
    try:
        rc = main(argv=["--quiet"])
    finally:
        hc.ALL_CHECKS = original

    assert rc == 1


def test_main_returns_zero_when_only_skips_and_passes() -> None:
    """Skips must not be counted as failures."""
    import alpamayo_r1.healthcheck as hc

    def p() -> CheckResult:
        return _ok("p", "ok")

    def s() -> CheckResult:
        return _skip("s", "no GPU")

    original = hc.ALL_CHECKS
    hc.ALL_CHECKS = (p, s)
    try:
        rc = main(argv=[])
    finally:
        hc.ALL_CHECKS = original

    assert rc == 0
