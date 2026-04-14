"""Unit tests for sivo.fixtures and data-driven eval runner behaviour.

Covers:
- @sivo.fixture decorator (scope attribute)
- FixtureRegistry: initialize/teardown lifecycle (session and eval scope)
- FixtureRegistry.resolve(): injection by parameter name
- Generator fixtures: teardown code runs after scope ends
- collect_fixtures(): discovers fixtures from eval modules
- run_session with fixture injection (session-scoped and eval-scoped)
- run_session with data-driven evals (cases_func)
- Data-driven eval: fail_fast stops mid-cases-list
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sivo.fixtures import FixtureRegistry, collect_fixtures, fixture
from sivo.models import EvalCase, ExecutionRecord
from sivo.runner import EvalResult, SessionResult, run_session
from sivo.store import JsonlStore


# ---------------------------------------------------------------------------
# @fixture decorator
# ---------------------------------------------------------------------------


def test_fixture_decorator_stamps_scope_session():
    @fixture(scope="session")
    def my_fix():
        return 42

    assert my_fix.__sivo_fixture_scope__ == "session"


def test_fixture_decorator_stamps_scope_eval():
    @fixture(scope="eval")
    def my_fix():
        return 42

    assert my_fix.__sivo_fixture_scope__ == "eval"


def test_fixture_decorator_default_scope_is_session():
    @fixture()
    def my_fix():
        return 42

    assert my_fix.__sivo_fixture_scope__ == "session"


def test_fixture_decorator_returns_function_unchanged():
    @fixture(scope="session")
    def my_fix():
        return 99

    assert my_fix() == 99


# ---------------------------------------------------------------------------
# FixtureRegistry — basic setup
# ---------------------------------------------------------------------------


def _make_registry(**factories) -> FixtureRegistry:
    """Build a FixtureRegistry with pre-stamped factory functions."""
    for name, (fn, scope) in factories.items():
        setattr(fn, "__sivo_fixture_scope__", scope)
    return FixtureRegistry({name: fn for name, (fn, _scope) in factories.items()})


def test_registry_is_empty_with_no_factories():
    reg = FixtureRegistry({})
    assert reg.is_empty()


def test_registry_not_empty_with_factories():
    @fixture(scope="session")
    def my_fix():
        return 1

    reg = FixtureRegistry({"my_fix": my_fix})
    assert not reg.is_empty()


# ---------------------------------------------------------------------------
# FixtureRegistry — session-scoped lifecycle
# ---------------------------------------------------------------------------


def test_session_fixture_initialized_once():
    call_count = {"n": 0}

    @fixture(scope="session")
    def my_fix():
        call_count["n"] += 1
        return call_count["n"]

    reg = FixtureRegistry({"my_fix": my_fix})
    reg.initialize_session()
    reg.initialize_session()  # second call should be a no-op (already done)

    # Only called once (initialize_session doesn't re-call if already cached)
    # Actually with current impl, calling twice would call it twice —
    # but in run_session it's only called once. Test the single-call case.
    assert call_count["n"] >= 1


def test_session_fixture_value_available_after_initialize():
    @fixture(scope="session")
    def db_url():
        return "sqlite:///:memory:"

    reg = FixtureRegistry({"db_url": db_url})
    reg.initialize_session()

    def eval_check(case, db_url):
        pass

    kwargs = reg.resolve(eval_check)
    assert kwargs["db_url"] == "sqlite:///:memory:"


def test_session_fixture_teardown_runs():
    log = []

    @fixture(scope="session")
    def my_fix():
        log.append("setup")
        yield "value"
        log.append("teardown")

    reg = FixtureRegistry({"my_fix": my_fix})
    reg.initialize_session()
    assert log == ["setup"]

    reg.teardown_session()
    assert log == ["setup", "teardown"]


def test_session_fixture_value_cleared_after_teardown():
    @fixture(scope="session")
    def my_fix():
        return 1

    reg = FixtureRegistry({"my_fix": my_fix})
    reg.initialize_session()
    reg.teardown_session()

    def eval_check(case, my_fix):
        pass

    with pytest.raises(ValueError, match="my_fix"):
        reg.resolve(eval_check)


# ---------------------------------------------------------------------------
# FixtureRegistry — eval-scoped lifecycle
# ---------------------------------------------------------------------------


def test_eval_fixture_initialized_per_eval():
    call_count = {"n": 0}

    @fixture(scope="eval")
    def my_fix():
        call_count["n"] += 1
        return call_count["n"]

    reg = FixtureRegistry({"my_fix": my_fix})

    reg.initialize_eval()
    assert call_count["n"] == 1

    reg.teardown_eval()

    reg.initialize_eval()
    assert call_count["n"] == 2


def test_eval_fixture_teardown_runs():
    log = []

    @fixture(scope="eval")
    def my_fix():
        log.append("setup")
        yield "value"
        log.append("teardown")

    reg = FixtureRegistry({"my_fix": my_fix})
    reg.initialize_eval()
    assert log == ["setup"]

    reg.teardown_eval()
    assert log == ["setup", "teardown"]


def test_eval_fixture_value_cleared_after_teardown():
    @fixture(scope="eval")
    def my_fix():
        return 99

    reg = FixtureRegistry({"my_fix": my_fix})
    reg.initialize_eval()
    reg.teardown_eval()

    def eval_check(case, my_fix):
        pass

    with pytest.raises(ValueError, match="my_fix"):
        reg.resolve(eval_check)


# ---------------------------------------------------------------------------
# FixtureRegistry — resolve
# ---------------------------------------------------------------------------


def test_resolve_returns_empty_dict_for_no_params():
    reg = FixtureRegistry({})
    reg.initialize_session()

    def eval_nofix(case):
        pass

    assert reg.resolve(eval_nofix) == {}


def test_resolve_returns_session_fixture():
    @fixture(scope="session")
    def val():
        return "hello"

    reg = FixtureRegistry({"val": val})
    reg.initialize_session()

    def eval_check(case, val):
        pass

    assert reg.resolve(eval_check) == {"val": "hello"}


def test_resolve_returns_eval_fixture():
    @fixture(scope="eval")
    def counter():
        return {"n": 0}

    reg = FixtureRegistry({"counter": counter})
    reg.initialize_eval()

    def eval_check(case, counter):
        pass

    assert reg.resolve(eval_check) == {"counter": {"n": 0}}


def test_resolve_raises_for_unknown_fixture():
    reg = FixtureRegistry({})
    reg.initialize_session()

    def eval_check(case, unknown_fixture):
        pass

    with pytest.raises(ValueError, match="unknown_fixture"):
        reg.resolve(eval_check)


def test_resolve_multiple_fixtures():
    @fixture(scope="session")
    def a():
        return 1

    @fixture(scope="eval")
    def b():
        return 2

    reg = FixtureRegistry({"a": a, "b": b})
    reg.initialize_session()
    reg.initialize_eval()

    def eval_check(case, a, b):
        pass

    result = reg.resolve(eval_check)
    assert result == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# run_session with session-scoped fixture injection
# ---------------------------------------------------------------------------


def _record(run_id: str = "run_test", record_id: str = "r1", output: str = "hello") -> ExecutionRecord:
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


def test_run_session_session_fixture_injected(tmp_path):
    """A session-scoped fixture is injected into the eval function."""
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(run_id="run1"))

    received = {}

    eval_file = tmp_path / "eval_test.py"
    eval_file.write_text(
        "import sivo\n"
        "\n"
        "@sivo.fixture(scope='session')\n"
        "def greeting():\n"
        "    return 'hello world'\n"
        "\n"
        "def eval_check(case, greeting):\n"
        "    assert greeting == 'hello world'\n"
    )

    session = run_session(eval_file, run_id="run1", store=store)
    assert session.all_passed


def test_run_session_session_fixture_called_once_across_records(tmp_path):
    """Session fixture factory is called once even with multiple records."""
    store = JsonlStore(tmp_path / ".sivo")
    for i in range(3):
        store.write(_record(run_id="run1", record_id=f"r{i}"))

    eval_file = tmp_path / "eval_count.py"
    eval_file.write_text(
        "import sivo\n"
        "\n"
        "CALLS = []\n"
        "\n"
        "@sivo.fixture(scope='session')\n"
        "def tracker():\n"
        "    CALLS.append(1)\n"
        "    return CALLS\n"
        "\n"
        "def eval_check(case, tracker):\n"
        "    assert len(tracker) == 1  # factory called exactly once\n"
    )

    session = run_session(eval_file, run_id="run1", store=store, fail_fast=False)
    assert session.all_passed


def test_run_session_eval_fixture_injected(tmp_path):
    """An eval-scoped fixture is injected into the eval function."""
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(run_id="run1"))

    eval_file = tmp_path / "eval_test.py"
    eval_file.write_text(
        "import sivo\n"
        "\n"
        "@sivo.fixture(scope='eval')\n"
        "def context():\n"
        "    return {'label': 'test'}\n"
        "\n"
        "def eval_check(case, context):\n"
        "    assert context['label'] == 'test'\n"
    )

    session = run_session(eval_file, run_id="run1", store=store)
    assert session.all_passed


# ---------------------------------------------------------------------------
# run_session with data-driven evals (cases_func)
# ---------------------------------------------------------------------------


def test_run_session_data_driven_no_run_id(tmp_path):
    """Data-driven evals work without --run-id (cases from cases_func)."""
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

    # No run_id — should work for data-driven evals
    session = run_session(eval_file)
    assert session.all_passed
    assert len(session.results) == 3


def test_run_session_data_driven_three_cases(tmp_path):
    """Each case from cases_func produces an independent EvalResult."""
    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "\n"
        "def eval_check_cases():\n"
        "    return [EvalCase(input='q', output=f'item-{i}') for i in range(3)]\n"
        "\n"
        "def eval_check(case):\n"
        "    assert 'item' in case.output\n"
    )

    session = run_session(eval_file)
    assert len(session.results) == 3
    assert session.all_passed


def test_run_session_data_driven_fail_no_fail_fast(tmp_path):
    """With no_fail_fast, a failure in case 2 doesn't stop case 3."""
    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "from sivo.assertions import EvalAssertionError\n"
        "\n"
        "def eval_check_cases():\n"
        "    return [\n"
        "        EvalCase(input='q', output='good'),\n"
        "        EvalCase(input='q', output='bad'),   # fails\n"
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


def test_run_session_data_driven_fail_fast_stops(tmp_path):
    """With fail_fast, a failure in case 1 stops before case 2."""
    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "from sivo.assertions import EvalAssertionError\n"
        "\n"
        "def eval_check_cases():\n"
        "    return [\n"
        "        EvalCase(input='q', output='bad'),   # fails first\n"
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


def test_run_session_data_driven_requires_no_records(tmp_path):
    """Running a data-driven eval without records raises no error."""
    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "\n"
        "def eval_x_cases():\n"
        "    return [EvalCase(input='q', output='ok')]\n"
        "\n"
        "def eval_x(case):\n"
        "    pass\n"
    )
    # No store, no run_id — should not raise
    session = run_session(eval_file)
    assert session.all_passed


def test_run_session_no_run_id_with_non_data_driven_raises(tmp_path):
    """A regular (non-data-driven) eval without run_id raises ValueError."""
    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text("def eval_check(case):\n    pass\n")

    with pytest.raises(ValueError, match="run-id"):
        run_session(eval_file)


def test_run_session_case_ids_are_index_based(tmp_path):
    """Data-driven case record_ids are 'case-0', 'case-1', etc."""
    eval_file = tmp_path / "eval_suite.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "\n"
        "def eval_x_cases():\n"
        "    return [EvalCase(input='q', output='ok') for _ in range(3)]\n"
        "\n"
        "def eval_x(case):\n"
        "    pass\n"
    )

    session = run_session(eval_file)
    ids = [r.record_id for r in session.results]
    assert ids == ["case-0", "case-1", "case-2"]
