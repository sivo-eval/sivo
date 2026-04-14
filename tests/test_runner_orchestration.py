"""Unit tests for sivo.runner — eval engine and orchestration layer.

These tests cover EvalEngine, get_response(), SessionResult, and run_session().
No LLM calls are made anywhere in this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sivo.assertions import EvalAssertionError, assert_contains
from sivo.models import EvalCase, ExecutionRecord
from sivo.runner import (
    EvalEngine,
    EvalResult,
    SessionResult,
    get_response,
    run_session,
)
from sivo.store import JsonlStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _case(output: str = "Hello world", **kwargs) -> EvalCase:
    return EvalCase(input="test input", output=output, **kwargs)


def _record(run_id: str = "run_test", record_id: str = "rec-1", output: str = "Hello world") -> ExecutionRecord:
    return ExecutionRecord(
        id=record_id,
        timestamp="2026-01-01T00:00:00+00:00",
        run_id=run_id,
        input="test input",
        output=output,
        model="claude-haiku-4-5",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0,
    )


def _write_eval_file(tmp_path: Path, name: str, content: str) -> Path:
    f = tmp_path / f"{name}.py"
    f.write_text(content)
    return f


def _write_jsonl(store: JsonlStore, run_id: str, records: list[ExecutionRecord]) -> None:
    """Write records to *store* under *run_id*, overriding each record's own run_id."""
    for r in records:
        store.write(r.model_copy(update={"run_id": run_id}))


# ---------------------------------------------------------------------------
# get_response
# ---------------------------------------------------------------------------


def test_get_response_returns_output():
    case = _case(output="Expected text")
    assert get_response(case) == "Expected text"


def test_get_response_empty_string():
    case = _case(output="")
    assert get_response(case) == ""


def test_get_response_whitespace():
    case = _case(output="  \n  ")
    assert get_response(case) == "  \n  "


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


def test_eval_result_passed():
    r = EvalResult(eval_name="eval_tone", record_id="rec-1", passed=True)
    assert r.passed is True
    assert r.error is None


def test_eval_result_failed_with_error():
    exc = EvalAssertionError("bad", assertion_type="assert_contains")
    r = EvalResult(eval_name="eval_tone", record_id="rec-1", passed=False, error=exc)
    assert r.passed is False
    assert r.error is exc


# ---------------------------------------------------------------------------
# SessionResult
# ---------------------------------------------------------------------------


def test_session_result_all_pass():
    s = SessionResult(run_id="run_x")
    s.results = [
        EvalResult("eval_a", "r1", True),
        EvalResult("eval_b", "r2", True),
    ]
    assert s.passed_count == 2
    assert s.failed_count == 0
    assert s.all_passed is True


def test_session_result_mixed():
    s = SessionResult(run_id="run_x")
    s.results = [
        EvalResult("eval_a", "r1", True),
        EvalResult("eval_b", "r2", False),
        EvalResult("eval_c", "r3", True),
    ]
    assert s.passed_count == 2
    assert s.failed_count == 1
    assert s.all_passed is False


def test_session_result_all_fail():
    s = SessionResult(run_id="run_x")
    s.results = [
        EvalResult("eval_a", "r1", False),
        EvalResult("eval_b", "r2", False),
    ]
    assert s.all_passed is False
    assert s.failed_count == 2


def test_session_result_empty():
    s = SessionResult(run_id="run_x")
    assert s.passed_count == 0
    assert s.failed_count == 0
    assert s.all_passed is True


# ---------------------------------------------------------------------------
# EvalEngine.run
# ---------------------------------------------------------------------------


def test_eval_engine_passes_on_no_assertion_raised():
    engine = EvalEngine()
    case = _case()

    def eval_ok(c):
        pass  # no assertion raised → pass

    result = engine.run(eval_ok, case, eval_name="eval_ok", record_id="rec-1")
    assert result.passed is True
    assert result.error is None
    assert result.eval_name == "eval_ok"
    assert result.record_id == "rec-1"


def test_eval_engine_fails_on_assertion_error():
    engine = EvalEngine()
    case = _case(output="wrong text")

    def eval_bad(c):
        assert_contains(c.output, "expected")

    result = engine.run(eval_bad, case, eval_name="eval_bad", record_id="rec-1")
    assert result.passed is False
    assert isinstance(result.error, EvalAssertionError)


def test_eval_engine_fails_on_standard_assertion_error():
    engine = EvalEngine()
    case = _case()

    def eval_assert(c):
        assert False, "custom failure"  # noqa: S101

    result = engine.run(eval_assert, case, eval_name="eval_assert", record_id="r1")
    assert result.passed is False
    assert isinstance(result.error, AssertionError)


def test_eval_engine_fails_on_unexpected_exception():
    engine = EvalEngine()
    case = _case()

    def eval_crashes(c):
        raise RuntimeError("unexpected crash")

    result = engine.run(eval_crashes, case, eval_name="eval_crash", record_id="r1")
    assert result.passed is False
    assert isinstance(result.error, RuntimeError)


def test_eval_engine_passes_case_to_function():
    engine = EvalEngine()
    case = _case(output="specific output")
    received = []

    def eval_capture(c):
        received.append(c.output)

    engine.run(eval_capture, case, eval_name="eval_capture", record_id="r1")
    assert received == ["specific output"]


# ---------------------------------------------------------------------------
# run_session
# ---------------------------------------------------------------------------


def test_run_session_all_pass(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    records = [
        _record(output="good response", record_id="r1"),
        _record(output="good answer", record_id="r2"),
    ]
    _write_jsonl(store, "run_test", records)

    eval_file = _write_eval_file(
        tmp_path,
        "eval_check",
        "from sivo.assertions import assert_contains\n"
        "def eval_check(case):\n"
        "    assert_contains(case.output, 'good')\n",
    )

    session = run_session(
        eval_file, run_id="run_test", store=store, fail_fast=False
    )

    assert session.all_passed
    assert session.passed_count == 2
    assert session.failed_count == 0
    assert len(session.results) == 2


def test_run_session_mixed_results(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    records = [
        _record(output="good response", record_id="r1"),
        _record(output="bad response", record_id="r2"),
        _record(output="good again", record_id="r3"),
    ]
    _write_jsonl(store, "run_test", records)

    eval_file = _write_eval_file(
        tmp_path,
        "eval_check",
        "from sivo.assertions import assert_contains\n"
        "def eval_check(case):\n"
        "    assert_contains(case.output, 'good')\n",
    )

    session = run_session(
        eval_file, run_id="run_test", store=store, fail_fast=False
    )

    assert session.all_passed is False
    assert session.passed_count == 2
    assert session.failed_count == 1


def test_run_session_fail_fast_stops_early(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    records = [
        _record(output="bad", record_id="r1"),
        _record(output="good", record_id="r2"),
        _record(output="good", record_id="r3"),
    ]
    _write_jsonl(store, "run_ff", records)

    eval_file = _write_eval_file(
        tmp_path,
        "eval_check",
        "from sivo.assertions import assert_contains\n"
        "def eval_check(case):\n"
        "    assert_contains(case.output, 'good')\n",
    )

    session = run_session(
        eval_file, run_id="run_ff", store=store, fail_fast=True
    )

    # Stopped after first failure — only 1 result
    assert len(session.results) == 1
    assert session.results[0].passed is False


def test_run_session_multiple_eval_functions(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    _write_jsonl(store, "run_multi", [_record(output="good and correct")])

    # Two eval files
    _write_eval_file(
        tmp_path,
        "eval_tone",
        "from sivo.assertions import assert_contains\n"
        "def eval_tone(case): assert_contains(case.output, 'good')\n",
    )
    _write_eval_file(
        tmp_path,
        "eval_accuracy",
        "from sivo.assertions import assert_contains\n"
        "def eval_accuracy(case): assert_contains(case.output, 'correct')\n",
    )

    session = run_session(
        tmp_path, run_id="run_multi", store=store, fail_fast=False
    )

    assert len(session.results) == 2  # 2 funcs × 1 record
    assert session.all_passed


def test_run_session_eval_filter(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    _write_jsonl(store, "run_filter", [_record(output="ok")])

    _write_eval_file(
        tmp_path, "eval_a", "def eval_a(case): pass\n"
    )
    _write_eval_file(
        tmp_path, "eval_b", "def eval_b(case): pass\n"
    )

    session = run_session(
        tmp_path, run_id="run_filter", store=store,
        eval_filter="eval_a", fail_fast=False,
    )

    assert len(session.results) == 1
    assert session.results[0].eval_name == "eval_a"


def test_run_session_raises_if_no_evals_found(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    _write_jsonl(store, "run_x", [_record()])

    with pytest.raises(ValueError, match="No eval functions found"):
        run_session(tmp_path, run_id="run_x", store=store)


def test_run_session_raises_if_no_records(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    _write_eval_file(tmp_path, "eval_a", "def eval_a(case): pass\n")

    with pytest.raises(ValueError, match="No records found"):
        run_session(tmp_path, run_id="no_such_run", store=store)


def test_run_session_eval_name_in_results(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    _write_jsonl(store, "run_names", [_record(output="x")])
    _write_eval_file(tmp_path, "eval_check", "def eval_check(case): pass\n")

    session = run_session(tmp_path, run_id="run_names", store=store)
    assert session.results[0].eval_name == "eval_check"


def test_run_session_record_id_in_results(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    rec = _record(record_id="specific-rec-id")
    store.write(rec)
    _write_eval_file(tmp_path, "eval_check", "def eval_check(case): pass\n")

    session = run_session(tmp_path, run_id="run_test", store=store)
    assert session.results[0].record_id == "specific-rec-id"


def test_run_session_get_response_in_eval(tmp_path):
    """Eval functions can use get_response(case) as well as case.output."""
    store = JsonlStore(tmp_path / ".sivo")
    _write_jsonl(store, "run_gr", [_record(output="target text")])

    eval_file = _write_eval_file(
        tmp_path,
        "eval_gr",
        "from sivo import get_response\n"
        "from sivo.assertions import assert_contains\n"
        "def eval_gr(case):\n"
        "    response = get_response(case)\n"
        "    assert_contains(response, 'target')\n",
    )

    session = run_session(eval_file, run_id="run_gr", store=store)
    assert session.all_passed
