"""Unit tests for sivo.replay.

Verifies that:
- parse_filters() correctly parses KEY=VALUE strings
- replay_session() loads records and runs evals without calling the LLM
- metadata_filter is forwarded to the store correctly
- --eval filter restricts which evals run

No LLM calls are made in this file (Anthropic client is never instantiated).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from sivo.models import ExecutionRecord
from sivo.replay import parse_filters, replay_session
from sivo.runner import EvalResult, SessionResult
from sivo.store import JsonlStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(
    run_id: str = "run_test",
    record_id: str = "rec-1",
    output: str = "Hello world",
    metadata: dict | None = None,
) -> ExecutionRecord:
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
        metadata=metadata or {},
    )


def _write_eval(tmp_path: Path, content: str, name: str = "eval_test") -> Path:
    f = tmp_path / f"{name}.py"
    f.write_text(content)
    return f


# ---------------------------------------------------------------------------
# parse_filters
# ---------------------------------------------------------------------------


def test_parse_filters_empty():
    assert parse_filters([]) == {}


def test_parse_filters_single():
    assert parse_filters(["env=prod"]) == {"env": "prod"}


def test_parse_filters_multiple():
    result = parse_filters(["env=prod", "region=us-east"])
    assert result == {"env": "prod", "region": "us-east"}


def test_parse_filters_value_with_equals():
    """Value itself may contain '=' — only the first = is the separator."""
    result = parse_filters(["key=a=b"])
    assert result == {"key": "a=b"}


def test_parse_filters_invalid_raises():
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_filters(["no-equals-sign"])


# ---------------------------------------------------------------------------
# replay_session — basic behaviour
# ---------------------------------------------------------------------------


def test_replay_session_returns_session_result(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(run_id="run1", record_id="r1", output="good"))

    _write_eval(tmp_path, "def eval_check(case):\n    pass\n")

    result = replay_session(tmp_path / "eval_test.py", run_id="run1", store=store)
    assert isinstance(result, SessionResult)


def test_replay_session_passes_clean_eval(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(run_id="run1", record_id="r1", output="good"))

    _write_eval(tmp_path, "def eval_check(case):\n    pass\n")

    session = replay_session(tmp_path / "eval_test.py", run_id="run1", store=store)
    assert session.all_passed is True
    assert session.passed_count == 1


def test_replay_session_fails_failing_eval(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(run_id="run1", record_id="r1", output="bad"))

    content = (
        "from sivo.assertions import EvalAssertionError\n"
        "def eval_check(case):\n"
        "    raise EvalAssertionError('bad output', assertion_type='assert_contains')\n"
    )
    _write_eval(tmp_path, content)

    session = replay_session(
        tmp_path / "eval_test.py",
        run_id="run1",
        store=store,
        fail_fast=False,
    )
    assert session.all_passed is False
    assert session.failed_count == 1


def test_replay_session_no_lm_calls(tmp_path):
    """Anthropic client must never be instantiated during replay."""
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(run_id="run1", record_id="r1"))

    _write_eval(tmp_path, "def eval_check(case):\n    pass\n")

    with patch("anthropic.Anthropic") as mock_sync, \
         patch("anthropic.AsyncAnthropic") as mock_async:
        replay_session(tmp_path / "eval_test.py", run_id="run1", store=store)
        mock_sync.assert_not_called()
        mock_async.assert_not_called()


# ---------------------------------------------------------------------------
# replay_session — eval_filter
# ---------------------------------------------------------------------------


def test_replay_session_eval_filter_restricts(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(run_id="run1", record_id="r1"))

    content = (
        "def eval_a(case):\n    pass\n"
        "def eval_b(case):\n    raise AssertionError('b always fails')\n"
    )
    _write_eval(tmp_path, content)

    # Only run eval_a — eval_b failure should not appear
    session = replay_session(
        tmp_path / "eval_test.py",
        run_id="run1",
        store=store,
        eval_filter="eval_a",
    )
    assert session.all_passed is True
    assert len(session.results) == 1
    assert session.results[0].eval_name == "eval_a"


def test_replay_session_eval_filter_unknown_raises(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(run_id="run1", record_id="r1"))

    _write_eval(tmp_path, "def eval_check(case):\n    pass\n")

    with pytest.raises(ValueError, match="No eval functions found"):
        replay_session(
            tmp_path / "eval_test.py",
            run_id="run1",
            store=store,
            eval_filter="eval_nonexistent",
        )


# ---------------------------------------------------------------------------
# replay_session — metadata_filter
# ---------------------------------------------------------------------------


def test_replay_session_metadata_filter_restricts(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(run_id="run1", record_id="r1", metadata={"env": "prod"}))
    store.write(_record(run_id="run1", record_id="r2", metadata={"env": "staging"}))

    _write_eval(tmp_path, "def eval_check(case):\n    pass\n")

    session = replay_session(
        tmp_path / "eval_test.py",
        run_id="run1",
        store=store,
        metadata_filter={"env": "prod"},
        fail_fast=False,
    )
    # Only the prod record should be replayed
    assert len(session.results) == 1
    assert session.results[0].record_id == "r1"


def test_replay_session_metadata_filter_empty_match_raises(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(run_id="run1", record_id="r1", metadata={"env": "prod"}))

    _write_eval(tmp_path, "def eval_check(case):\n    pass\n")

    # Filter that matches nothing
    with pytest.raises(ValueError, match="No records found"):
        replay_session(
            tmp_path / "eval_test.py",
            run_id="run1",
            store=store,
            metadata_filter={"env": "nonexistent"},
        )


# ---------------------------------------------------------------------------
# replay_session — on_result callback
# ---------------------------------------------------------------------------


def test_replay_session_on_result_called(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(run_id="run1", record_id="r1"))
    store.write(_record(run_id="run1", record_id="r2"))

    _write_eval(tmp_path, "def eval_check(case):\n    pass\n")

    collected: list[EvalResult] = []
    replay_session(
        tmp_path / "eval_test.py",
        run_id="run1",
        store=store,
        on_result=collected.append,
        fail_fast=False,
    )
    assert len(collected) == 2


# ---------------------------------------------------------------------------
# replay_session — run_id not found
# ---------------------------------------------------------------------------


def test_replay_session_unknown_run_id_raises(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")

    _write_eval(tmp_path, "def eval_check(case):\n    pass\n")

    with pytest.raises(ValueError, match="No records found"):
        replay_session(
            tmp_path / "eval_test.py",
            run_id="nonexistent",
            store=store,
        )
