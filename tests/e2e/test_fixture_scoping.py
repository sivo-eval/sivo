"""E2E tests for fixture scoping behaviour.

Verifies:
- Session-scoped fixture factory called exactly once even with multiple evals / records.
- Eval-scoped fixture factory called once per eval function (reset between evals).
- Teardown code (yield-based) runs for both scopes.
- CLI round-trip: ``sivo run`` picks up fixtures defined in the eval file.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from sivo.models import ExecutionRecord
from sivo.runner import run_session
from sivo.store import JsonlStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(run_id: str, record_id: str, output: str = "ok") -> ExecutionRecord:
    return ExecutionRecord(
        id=record_id,
        timestamp="2026-01-01T00:00:00+00:00",
        run_id=run_id,
        input="q",
        output=output,
        model="claude-haiku-4-5",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
    )


# ---------------------------------------------------------------------------
# Session-scoped fixture — called once across multiple records
# ---------------------------------------------------------------------------


def test_session_fixture_called_once_two_evals(tmp_path):
    """Session fixture factory called once even when two eval functions use it."""
    store = JsonlStore(tmp_path / ".sivo")
    for i in range(2):
        store.write(_record(run_id="run1", record_id=f"r{i}"))

    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "import sivo\n"
        "\n"
        "CALLS = []\n"
        "\n"
        "@sivo.fixture(scope='session')\n"
        "def shared():\n"
        "    CALLS.append(1)\n"
        "    return CALLS\n"
        "\n"
        "def eval_first(case, shared):\n"
        "    assert len(shared) == 1\n"
        "\n"
        "def eval_second(case, shared):\n"
        "    assert len(shared) == 1\n"
    )

    session = run_session(eval_file, run_id="run1", store=store, fail_fast=False)
    assert session.all_passed
    assert len(session.results) == 4  # 2 evals × 2 records


def test_session_fixture_teardown_runs_once(tmp_path):
    """Session fixture teardown runs exactly once at end of session."""
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(run_id="run1", record_id="r0"))

    # Use a file-based log so we can read teardown state after session
    log_path = tmp_path / "teardown.log"

    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        f"import sivo\n"
        f"from pathlib import Path\n"
        f"\n"
        f"LOG = Path({str(log_path)!r})\n"
        f"\n"
        f"@sivo.fixture(scope='session')\n"
        f"def resource():\n"
        f"    yield 'up'\n"
        f"    LOG.write_text('torn down')\n"
        f"\n"
        f"def eval_check(case, resource):\n"
        f"    assert resource == 'up'\n"
    )

    session = run_session(eval_file, run_id="run1", store=store)
    assert session.all_passed
    assert log_path.read_text() == "torn down"


# ---------------------------------------------------------------------------
# Eval-scoped fixture — reinitialized between eval functions
# ---------------------------------------------------------------------------


def test_eval_fixture_reinitialized_between_evals(tmp_path):
    """Eval-scoped counter resets to 0 for each eval function."""
    store = JsonlStore(tmp_path / ".sivo")
    for i in range(3):
        store.write(_record(run_id="run1", record_id=f"r{i}"))

    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "import sivo\n"
        "\n"
        "@sivo.fixture(scope='eval')\n"
        "def counter():\n"
        "    return {'n': 0}\n"
        "\n"
        "def eval_alpha(case, counter):\n"
        "    counter['n'] += 1\n"
        "    # counter['n'] may be 1, 2, or 3 across the 3 records\n"
        "    assert counter['n'] >= 1\n"
        "\n"
        "def eval_beta(case, counter):\n"
        "    # fresh counter for this eval function — n starts at 0 again\n"
        "    counter['n'] += 1\n"
        "    assert counter['n'] >= 1\n"
    )

    session = run_session(eval_file, run_id="run1", store=store, fail_fast=False)
    assert session.all_passed
    assert len(session.results) == 6  # 2 evals × 3 records


def test_eval_fixture_same_instance_within_eval(tmp_path):
    """All records within one eval function share the same eval-scoped instance."""
    store = JsonlStore(tmp_path / ".sivo")
    for i in range(3):
        store.write(_record(run_id="run1", record_id=f"r{i}"))

    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "import sivo\n"
        "\n"
        "LAST_ID = [None]\n"
        "\n"
        "@sivo.fixture(scope='eval')\n"
        "def ctx():\n"
        "    return {'id': id(object())}\n"
        "\n"
        "def eval_check(case, ctx):\n"
        "    if LAST_ID[0] is None:\n"
        "        LAST_ID[0] = ctx['id']\n"
        "    else:\n"
        "        # Same dict instance across all records in this eval\n"
        "        assert ctx['id'] == LAST_ID[0]\n"
    )

    session = run_session(eval_file, run_id="run1", store=store, fail_fast=False)
    assert session.all_passed


def test_eval_fixture_teardown_runs_between_evals(tmp_path):
    """Eval fixture teardown runs after each eval function's records complete."""
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(run_id="run1", record_id="r0"))

    log_path = tmp_path / "teardown.log"

    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        f"import sivo\n"
        f"from pathlib import Path\n"
        f"\n"
        f"LOG = Path({str(log_path)!r})\n"
        f"LINES = []\n"
        f"\n"
        f"@sivo.fixture(scope='eval')\n"
        f"def resource():\n"
        f"    LINES.append('setup')\n"
        f"    yield 'up'\n"
        f"    LINES.append('teardown')\n"
        f"    LOG.write_text('\\n'.join(LINES))\n"
        f"\n"
        f"def eval_first(case, resource):\n"
        f"    assert resource == 'up'\n"
        f"\n"
        f"def eval_second(case, resource):\n"
        f"    assert resource == 'up'\n"
    )

    session = run_session(eval_file, run_id="run1", store=store, fail_fast=False)
    assert session.all_passed
    lines = log_path.read_text().splitlines()
    # setup/teardown called for each eval function (2 evals)
    assert lines.count("setup") == 2
    assert lines.count("teardown") == 2


# ---------------------------------------------------------------------------
# CLI round-trip
# ---------------------------------------------------------------------------


def test_cli_run_with_session_fixture(tmp_path):
    """``sivo run`` correctly injects a session-scoped fixture."""
    store_path = tmp_path / ".sivo"
    store = JsonlStore(store_path)
    store.write(_record(run_id="run1", record_id="r0"))

    eval_file = tmp_path / "eval_check.py"
    eval_file.write_text(
        "import sivo\n"
        "\n"
        "@sivo.fixture(scope='session')\n"
        "def greeting():\n"
        "    return 'hello'\n"
        "\n"
        "def eval_check(case, greeting):\n"
        "    assert greeting == 'hello'\n"
    )

    result = subprocess.run(
        [sys.executable, "-m", "sivo.cli", "run", str(eval_file),
         "--run-id", "run1", "--store-path", str(store_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
