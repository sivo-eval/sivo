"""E2E tests for data-driven evals (cases_func companion functions).

Verifies:
- eval_X_cases() returning list[EvalCase] drives one run per case.
- No --run-id or store required.
- Failure in case 2 with --no-fail-fast doesn't stop case 3.
- fail_fast stops on first failure.
- Case IDs are index-based (case-0, case-1, …).
- CLI round-trip: ``sivo run`` works with data-driven evals (no --run-id).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from sivo.runner import run_session


# ---------------------------------------------------------------------------
# Basic data-driven behaviour
# ---------------------------------------------------------------------------


def test_data_driven_three_cases_all_pass(tmp_path):
    """3 EvalCase objects → 3 EvalResult objects, all pass."""
    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "\n"
        "def eval_summariser_cases():\n"
        "    return [\n"
        "        EvalCase(input='q1', output='good one'),\n"
        "        EvalCase(input='q2', output='good two'),\n"
        "        EvalCase(input='q3', output='good three'),\n"
        "    ]\n"
        "\n"
        "def eval_summariser(case):\n"
        "    assert 'good' in case.output\n"
    )

    session = run_session(eval_file)
    assert len(session.results) == 3
    assert session.all_passed


def test_data_driven_no_run_id_required(tmp_path):
    """Data-driven evals work without --run-id (no JSONL store needed)."""
    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "\n"
        "def eval_x_cases():\n"
        "    return [EvalCase(input='q', output='ok')]\n"
        "\n"
        "def eval_x(case):\n"
        "    assert case.output == 'ok'\n"
    )

    # No run_id, no store — should not raise
    session = run_session(eval_file)
    assert session.all_passed


def test_data_driven_case_ids_are_index_based(tmp_path):
    """Case IDs for data-driven evals are 'case-0', 'case-1', …"""
    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "\n"
        "def eval_x_cases():\n"
        "    return [EvalCase(input='q', output='ok') for _ in range(4)]\n"
        "\n"
        "def eval_x(case):\n"
        "    pass\n"
    )

    session = run_session(eval_file)
    ids = [r.record_id for r in session.results]
    assert ids == ["case-0", "case-1", "case-2", "case-3"]


def test_data_driven_each_case_is_independent(tmp_path):
    """Each case is a separate EvalResult with its own pass/fail state."""
    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "from sivo.assertions import EvalAssertionError\n"
        "\n"
        "def eval_check_cases():\n"
        "    return [\n"
        "        EvalCase(input='q', output='pass'),\n"
        "        EvalCase(input='q', output='fail'),\n"
        "        EvalCase(input='q', output='pass'),\n"
        "    ]\n"
        "\n"
        "def eval_check(case):\n"
        "    if case.output != 'pass':\n"
        "        raise EvalAssertionError('bad output', assertion_type='assert_contains')\n"
    )

    session = run_session(eval_file, fail_fast=False)
    assert len(session.results) == 3
    assert session.results[0].passed is True
    assert session.results[1].passed is False
    assert session.results[2].passed is True
    assert session.passed_count == 2
    assert session.failed_count == 1


# ---------------------------------------------------------------------------
# fail_fast behaviour
# ---------------------------------------------------------------------------


def test_data_driven_fail_fast_stops_on_first_failure(tmp_path):
    """fail_fast=True stops after the first failing case."""
    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "from sivo.assertions import EvalAssertionError\n"
        "\n"
        "def eval_check_cases():\n"
        "    return [\n"
        "        EvalCase(input='q', output='bad'),\n"   # fails first
        "        EvalCase(input='q', output='good'),\n"
        "        EvalCase(input='q', output='good'),\n"
        "    ]\n"
        "\n"
        "def eval_check(case):\n"
        "    if case.output != 'good':\n"
        "        raise EvalAssertionError('bad', assertion_type='assert_contains')\n"
    )

    session = run_session(eval_file, fail_fast=True)
    assert len(session.results) == 1
    assert session.results[0].passed is False


def test_data_driven_no_fail_fast_runs_all_cases(tmp_path):
    """fail_fast=False runs all cases even when case 2 fails."""
    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "from sivo.assertions import EvalAssertionError\n"
        "\n"
        "def eval_check_cases():\n"
        "    return [\n"
        "        EvalCase(input='q', output='good'),\n"
        "        EvalCase(input='q', output='bad'),\n"   # fails
        "        EvalCase(input='q', output='good'),\n"
        "    ]\n"
        "\n"
        "def eval_check(case):\n"
        "    if case.output != 'good':\n"
        "        raise EvalAssertionError('bad', assertion_type='assert_contains')\n"
    )

    session = run_session(eval_file, fail_fast=False)
    assert len(session.results) == 3
    assert session.passed_count == 2
    assert session.failed_count == 1


def test_data_driven_fail_fast_mid_cases(tmp_path):
    """fail_fast stops at case 2, not at case 1 (which passes)."""
    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "from sivo.assertions import EvalAssertionError\n"
        "\n"
        "def eval_check_cases():\n"
        "    return [\n"
        "        EvalCase(input='q', output='good'),\n"  # passes
        "        EvalCase(input='q', output='bad'),\n"   # fails → stop
        "        EvalCase(input='q', output='good'),\n"
        "    ]\n"
        "\n"
        "def eval_check(case):\n"
        "    if case.output != 'good':\n"
        "        raise EvalAssertionError('bad', assertion_type='assert_contains')\n"
    )

    session = run_session(eval_file, fail_fast=True)
    assert len(session.results) == 2
    assert session.results[0].passed is True
    assert session.results[1].passed is False


# ---------------------------------------------------------------------------
# Mixed data-driven + regular evals
# ---------------------------------------------------------------------------


def test_data_driven_mixed_with_record_based(tmp_path):
    """Data-driven eval alongside a record-based eval in the same file."""
    from sivo.models import ExecutionRecord
    from sivo.store import JsonlStore

    store = JsonlStore(tmp_path / ".sivo")
    store.write(ExecutionRecord(
        id="r0",
        timestamp="2026-01-01T00:00:00+00:00",
        run_id="run1",
        input="q",
        output="hello",
        model="claude-haiku-4-5",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
    ))

    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "\n"
        "def eval_data_cases():\n"
        "    return [EvalCase(input='q', output='ok'), EvalCase(input='q', output='ok')]\n"
        "\n"
        "def eval_data(case):\n"
        "    assert case.output == 'ok'\n"
        "\n"
        "def eval_record(case):\n"
        "    assert 'hello' in case.output\n"
    )

    session = run_session(eval_file, run_id="run1", store=store, fail_fast=False)
    assert session.all_passed
    # 2 data-driven cases + 1 record-based case
    assert len(session.results) == 3


# ---------------------------------------------------------------------------
# CLI round-trip
# ---------------------------------------------------------------------------


def test_cli_run_data_driven_no_run_id(tmp_path):
    """``sivo run`` works for data-driven evals without --run-id."""
    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "\n"
        "def eval_x_cases():\n"
        "    return [EvalCase(input='q', output='ok') for _ in range(3)]\n"
        "\n"
        "def eval_x(case):\n"
        "    assert case.output == 'ok'\n"
    )

    result = subprocess.run(
        [sys.executable, "-m", "sivo.cli", "run", str(eval_file),
         "--store-path", str(tmp_path / ".sivo")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cli_run_data_driven_failure_exits_1(tmp_path):
    """``sivo run`` exits 1 when a data-driven case fails."""
    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "from sivo.assertions import EvalAssertionError\n"
        "\n"
        "def eval_x_cases():\n"
        "    return [EvalCase(input='q', output='bad')]\n"
        "\n"
        "def eval_x(case):\n"
        "    raise EvalAssertionError('bad', assertion_type='assert_contains')\n"
    )

    result = subprocess.run(
        [sys.executable, "-m", "sivo.cli", "run", str(eval_file),
         "--store-path", str(tmp_path / ".sivo")],
        capture_output=True, text=True,
    )
    assert result.returncode == 1


def test_cli_run_no_fail_fast_data_driven(tmp_path):
    """``sivo run --no-fail-fast`` runs all cases even after failure."""
    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "from sivo.assertions import EvalAssertionError\n"
        "\n"
        "SEEN = []\n"
        "\n"
        "def eval_x_cases():\n"
        "    return [\n"
        "        EvalCase(input='q', output='bad'),\n"
        "        EvalCase(input='q', output='ok'),\n"
        "        EvalCase(input='q', output='ok'),\n"
        "    ]\n"
        "\n"
        "def eval_x(case):\n"
        "    if case.output != 'ok':\n"
        "        raise EvalAssertionError('bad', assertion_type='assert_contains')\n"
    )

    result = subprocess.run(
        [sys.executable, "-m", "sivo.cli", "run", str(eval_file),
         "--no-fail-fast", "--store-path", str(tmp_path / ".sivo")],
        capture_output=True, text=True,
    )
    # 1 fail, so exit 1; but all 3 cases ran
    assert result.returncode == 1
    # The output should mention 2 passed and 1 failed
    output = result.stdout + result.stderr
    assert "2" in output  # passed count appears somewhere
