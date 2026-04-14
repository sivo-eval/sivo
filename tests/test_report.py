"""Unit tests for sivo.report and the Phase 5 runner additions.

All tests use an in-memory Rich console so no TTY is needed.
"""

from __future__ import annotations

import json
from io import StringIO

import pytest
from rich.console import Console

from sivo.assertions import EvalAssertionError, FlakyEvalError
from sivo.models import JudgeVerdict
from sivo.report import print_receipt, print_result, print_session
from sivo.runner import EvalResult, SessionResult


# ---------------------------------------------------------------------------
# Console helper
# ---------------------------------------------------------------------------


def _capture_console() -> Console:
    """Return a Rich console that writes to a StringIO buffer."""
    return Console(file=StringIO(), highlight=False, markup=False, no_color=True)


def _text(console: Console) -> str:
    return console.file.getvalue()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# EvalResult — flaky field
# ---------------------------------------------------------------------------


def test_eval_result_default_not_flaky():
    r = EvalResult(eval_name="eval_x", record_id="r1", passed=True)
    assert r.flaky is False


def test_eval_result_flaky_flag():
    r = EvalResult(eval_name="eval_x", record_id="r1", passed=True, flaky=True)
    assert r.flaky is True
    assert r.passed is True  # flaky still "passes" by default


# ---------------------------------------------------------------------------
# SessionResult — counts with flaky
# ---------------------------------------------------------------------------


def test_session_passed_count_excludes_flaky():
    s = SessionResult(run_id="r")
    s.results = [
        EvalResult("a", "r1", True),
        EvalResult("b", "r2", True, flaky=True),
        EvalResult("c", "r3", False),
    ]
    assert s.passed_count == 1   # only the clean pass
    assert s.flaky_count == 1
    assert s.failed_count == 1


def test_session_all_passed_true_with_flaky():
    """Flaky results do not cause all_passed to be False."""
    s = SessionResult(run_id="r")
    s.results = [
        EvalResult("a", "r1", True),
        EvalResult("b", "r2", True, flaky=True),
    ]
    assert s.all_passed is True


def test_session_all_passed_false_with_failure():
    s = SessionResult(run_id="r")
    s.results = [
        EvalResult("a", "r1", True),
        EvalResult("b", "r2", False),
    ]
    assert s.all_passed is False


def test_session_cost_fields():
    s = SessionResult(
        run_id="r",
        total_input_tokens=100,
        total_output_tokens=50,
        total_cost_usd=0.00042,
        cost_by_eval={"eval_a": 0.00021, "eval_b": 0.00021},
    )
    assert s.total_input_tokens == 100
    assert s.total_output_tokens == 50
    assert s.total_cost_usd == pytest.approx(0.00042)
    assert s.cost_by_eval["eval_a"] == pytest.approx(0.00021)


# ---------------------------------------------------------------------------
# print_result — verbosity 0
# ---------------------------------------------------------------------------


def test_print_result_pass_shows_pass():
    con = _capture_console()
    r = EvalResult("eval_tone", "rec-0", True)
    print_result(r, verbose=0, console=con)
    assert "PASS" in _text(con)
    assert "eval_tone" in _text(con)


def test_print_result_fail_shows_fail():
    con = _capture_console()
    err = EvalAssertionError("text lacks 'good'", assertion_type="assert_contains")
    r = EvalResult("eval_check", "rec-1", False, error=err)
    print_result(r, verbose=0, console=con)
    out = _text(con)
    assert "FAIL" in out
    assert "eval_check" in out


def test_print_result_fail_shows_first_line_of_error_at_v0():
    con = _capture_console()
    err = EvalAssertionError(
        "line one\nline two\nline three", assertion_type="assert_contains"
    )
    r = EvalResult("eval_x", "rec-0", False, error=err)
    print_result(r, verbose=0, console=con)
    out = _text(con)
    assert "line one" in out
    assert "line two" not in out


def test_print_result_flaky_shows_flaky():
    con = _capture_console()
    err = FlakyEvalError("split result")
    r = EvalResult("eval_flaky", "rec-2", True, flaky=True, error=err)
    print_result(r, verbose=0, console=con)
    assert "FLAKY" in _text(con)


def test_print_result_record_id_in_output():
    con = _capture_console()
    r = EvalResult("eval_tone", "specific-record-id", True)
    print_result(r, verbose=0, console=con)
    assert "specific-record-id" in _text(con)


# ---------------------------------------------------------------------------
# print_result — verbosity 1 (-v)
# ---------------------------------------------------------------------------


def _make_judge_result(passed: bool = False) -> EvalResult:
    verdict = JudgeVerdict(
        passed=passed,
        reason="The response is too casual.",
        evidence='"hey what is up"',
        suggestion="Use formal language.",
    )
    err = EvalAssertionError(
        f"Judge failed.\nReason: {verdict.reason}",
        assertion_type="assert_judge",
        expected="pass rubric 'tone'",
        actual=verdict,
    )
    return EvalResult("eval_tone", "rec-1", False, error=err)


def test_print_result_v1_shows_evidence():
    con = _capture_console()
    r = _make_judge_result()
    print_result(r, verbose=1, console=con)
    out = _text(con)
    assert "Evidence:" in out
    assert '"hey what is up"' in out


def test_print_result_v1_shows_reason():
    con = _capture_console()
    r = _make_judge_result()
    print_result(r, verbose=1, console=con)
    assert "Reason:" in _text(con)
    assert "The response is too casual." in _text(con)


def test_print_result_v1_shows_suggestion():
    con = _capture_console()
    r = _make_judge_result()
    print_result(r, verbose=1, console=con)
    assert "Suggestion:" in _text(con)
    assert "Use formal language." in _text(con)


def test_print_result_v1_non_judge_shows_full_error():
    con = _capture_console()
    err = EvalAssertionError(
        "line one\nline two",
        assertion_type="assert_contains",
    )
    r = EvalResult("eval_x", "rec-0", False, error=err)
    print_result(r, verbose=1, console=con)
    out = _text(con)
    assert "line one" in out
    assert "line two" in out


# ---------------------------------------------------------------------------
# print_result — verbosity 2 (-vv)
# ---------------------------------------------------------------------------


def test_print_result_v2_shows_judge_json():
    con = _capture_console()
    r = _make_judge_result()
    print_result(r, verbose=2, console=con)
    out = _text(con)
    # JSON keys from JudgeVerdict
    assert '"passed"' in out
    assert '"reason"' in out
    assert '"evidence"' in out
    assert '"suggestion"' in out


def test_print_result_v2_json_is_valid():
    con = _capture_console()
    r = _make_judge_result()
    print_result(r, verbose=2, console=con)
    out = _text(con)
    # Extract JSON block (everything between first { and last })
    start = out.index("{")
    end = out.rindex("}") + 1
    parsed = json.loads(out[start:end])
    assert parsed["passed"] is False
    assert "reason" in parsed


# ---------------------------------------------------------------------------
# print_receipt
# ---------------------------------------------------------------------------


def _make_session(*, passed=2, failed=1, flaky=1) -> SessionResult:
    s = SessionResult(
        run_id="run_receipt_test",
        total_input_tokens=300,
        total_output_tokens=150,
        total_cost_usd=0.000480,
        cost_by_eval={"eval_a": 0.000240, "eval_b": 0.000240},
    )
    for i in range(passed):
        s.results.append(EvalResult(f"eval_pass_{i}", f"r{i}", True))
    for i in range(failed):
        s.results.append(
            EvalResult(
                f"eval_fail_{i}",
                f"rf{i}",
                False,
                error=EvalAssertionError("bad", assertion_type="assert_contains"),
            )
        )
    for i in range(flaky):
        s.results.append(
            EvalResult(f"eval_flaky_{i}", f"rfl{i}", True, flaky=True)
        )
    return s


def test_receipt_contains_run_id():
    con = _capture_console()
    s = _make_session()
    print_receipt(s, console=con)
    assert "run_receipt_test" in _text(con)


def test_receipt_contains_passed_count():
    con = _capture_console()
    s = _make_session(passed=3, failed=0, flaky=0)
    print_receipt(s, console=con)
    assert "3 passed" in _text(con)


def test_receipt_contains_failed_count():
    con = _capture_console()
    s = _make_session(passed=1, failed=2, flaky=0)
    print_receipt(s, console=con)
    assert "2 failed" in _text(con)


def test_receipt_contains_flaky_count():
    con = _capture_console()
    s = _make_session(passed=1, failed=0, flaky=2)
    print_receipt(s, console=con)
    assert "2 flaky" in _text(con)


def test_receipt_no_failed_line_omitted():
    """When there are no failures the '0 failed' string should not appear."""
    con = _capture_console()
    s = _make_session(passed=3, failed=0, flaky=0)
    print_receipt(s, console=con)
    assert "failed" not in _text(con)


def test_receipt_no_flaky_line_omitted():
    con = _capture_console()
    s = _make_session(passed=3, failed=0, flaky=0)
    print_receipt(s, console=con)
    assert "flaky" not in _text(con)


def test_receipt_token_totals():
    con = _capture_console()
    s = _make_session()
    print_receipt(s, console=con)
    out = _text(con)
    assert "300" in out
    assert "150" in out


def test_receipt_cost_present():
    con = _capture_console()
    s = _make_session()
    print_receipt(s, console=con)
    assert "cost:" in _text(con)
    assert "0.000480" in _text(con)


def test_receipt_cost_breakdown_multi_eval():
    con = _capture_console()
    s = _make_session()  # has eval_a and eval_b in cost_by_eval
    print_receipt(s, console=con)
    out = _text(con)
    assert "eval_a" in out
    assert "eval_b" in out


# ---------------------------------------------------------------------------
# print_session
# ---------------------------------------------------------------------------


def test_print_session_includes_all_results():
    con = _capture_console()
    s = SessionResult(run_id="run_x")
    s.results = [
        EvalResult("eval_a", "r1", True),
        EvalResult("eval_b", "r2", False,
                   error=EvalAssertionError("bad", assertion_type="assert_contains")),
    ]
    print_session(s, verbose=0, console=con)
    out = _text(con)
    assert "eval_a" in out
    assert "eval_b" in out
    assert "run_x" in out


# ---------------------------------------------------------------------------
# FlakyEvalError
# ---------------------------------------------------------------------------


def test_flaky_eval_error_is_exception():
    with pytest.raises(FlakyEvalError):
        raise FlakyEvalError("split verdict")


def test_flaky_eval_error_not_assertion_error():
    assert not issubclass(FlakyEvalError, AssertionError)
