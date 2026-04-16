"""Unit tests for sivo.repl — headless REPL command dispatch and hot-swap.

All tests use an injected ``input_fn`` (a scripted iterator) and a Rich
console backed by ``StringIO`` so no real terminal is required.
"""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from sivo.assertions import EvalAssertionError, FlakyEvalError
from sivo.models import EvalCase, JudgeVerdict
from sivo.repl import PdbLlmSession, make_pdb_hook
from sivo.runner import EvalEngine, EvalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _console() -> Console:
    return Console(file=StringIO(), highlight=False, markup=False, no_color=True)


def _text(con: Console) -> str:
    return con.file.getvalue()  # type: ignore[union-attr]


def _case(
    output: str = "some output",
    system_prompt: str | None = "You are helpful.",
    input: str = "test input",
) -> EvalCase:
    return EvalCase(input=input, output=output, system_prompt=system_prompt)


def _fail_result(eval_name: str = "eval_x", record_id: str = "r1") -> EvalResult:
    err = EvalAssertionError("text missing 'expected'", assertion_type="assert_contains")
    return EvalResult(eval_name=eval_name, record_id=record_id, passed=False, error=err)


def _judge_result(passed: bool = False) -> EvalResult:
    verdict = JudgeVerdict(
        passed=passed,
        reason="Response is too casual.",
        evidence='"hey what is up"',
        suggestion="Use formal language.",
    )
    err = EvalAssertionError(
        "Judge assertion failed.\nReason: Response is too casual.",
        assertion_type="assert_judge",
        expected="pass rubric 'tone'",
        actual=verdict,
    )
    return EvalResult("eval_tone", "r1", False, error=err)


def _session(
    commands: list[str],
    case: EvalCase | None = None,
    result: EvalResult | None = None,
    eval_func=None,
    provider=None,
) -> tuple[PdbLlmSession, Console]:
    con = _console()
    inputs = iter(commands)
    if case is None:
        case = _case()
    if result is None:
        result = _fail_result()
    if eval_func is None:
        eval_func = lambda c: None  # noqa: E731  always passes
    sess = PdbLlmSession(
        case=case,
        result=result,
        eval_func=eval_func,
        console=con,
        input_fn=lambda: next(inputs),
        provider=provider,
    )
    return sess, con


# ---------------------------------------------------------------------------
# Terminal commands — basic dispatch
# ---------------------------------------------------------------------------


def test_skip_returns_skip():
    sess, _ = _session(["skip"])
    action, _ = sess.run()
    assert action == "skip"


def test_continue_returns_continue():
    sess, _ = _session(["continue"])
    action, _ = sess.run()
    assert action == "continue"


def test_abort_returns_abort():
    sess, _ = _session(["abort"])
    action, _ = sess.run()
    assert action == "abort"


def test_eof_returns_abort():
    """EOFError from input_fn is treated as abort."""
    def _raise():
        raise EOFError

    con = _console()
    sess = PdbLlmSession(
        case=_case(),
        result=_fail_result(),
        eval_func=lambda c: None,
        console=con,
        input_fn=_raise,
    )
    action, _ = sess.run()
    assert action == "abort"


def test_keyboard_interrupt_returns_abort():
    """KeyboardInterrupt is also treated as abort."""
    def _raise():
        raise KeyboardInterrupt

    con = _console()
    sess = PdbLlmSession(
        case=_case(),
        result=_fail_result(),
        eval_func=lambda c: None,
        console=con,
        input_fn=_raise,
    )
    action, _ = sess.run()
    assert action == "abort"


def test_empty_line_continues_loop():
    """An empty line should not exit the REPL — just loop again."""
    sess, _ = _session(["", "skip"])
    action, _ = sess.run()
    assert action == "skip"


def test_unknown_command_shows_help():
    sess, con = _session(["nonsense", "abort"])
    sess.run()
    out = _text(con)
    assert "Unknown command" in out
    assert "nonsense" in out
    assert "Commands:" in out


def test_banner_shows_fail_label():
    sess, con = _session(["skip"])
    sess.run()
    assert "FAIL" in _text(con)


def test_banner_shows_eval_name():
    sess, con = _session(["skip"], result=_fail_result(eval_name="eval_my_check"))
    sess.run()
    assert "eval_my_check" in _text(con)


def test_banner_shows_record_id():
    sess, con = _session(["skip"], result=_fail_result(record_id="rec-xyz"))
    sess.run()
    assert "rec-xyz" in _text(con)


def test_banner_shows_first_error_line():
    sess, con = _session(["skip"])
    sess.run()
    assert "text missing 'expected'" in _text(con)


def test_banner_shows_flaky_label():
    flaky_result = EvalResult(
        "eval_x", "r1", True, flaky=True,
        error=FlakyEvalError("split verdict"),
    )
    sess, con = _session(["skip"], result=flaky_result)
    sess.run()
    assert "FLAKY" in _text(con)


# ---------------------------------------------------------------------------
# inspect command
# ---------------------------------------------------------------------------


def test_inspect_shows_input():
    sess, con = _session(["inspect", "skip"], case=_case(input="What is 2+2?"))
    sess.run()
    assert "What is 2+2?" in _text(con)


def test_inspect_shows_output():
    sess, con = _session(["inspect", "skip"], case=_case(output="the answer is 4"))
    sess.run()
    assert "the answer is 4" in _text(con)


def test_inspect_shows_system_prompt():
    sess, con = _session(
        ["inspect", "skip"],
        case=_case(system_prompt="Be concise."),
    )
    sess.run()
    assert "Be concise." in _text(con)


def test_inspect_shows_none_system_prompt():
    sess, con = _session(
        ["inspect", "skip"],
        case=EvalCase(input="q", output="a", system_prompt=None),
    )
    sess.run()
    assert "system_prompt: None" in _text(con)


def test_inspect_shows_judge_verdict_fields():
    sess, con = _session(["inspect", "skip"], result=_judge_result())
    sess.run()
    out = _text(con)
    assert "judge_verdict:" in out
    assert "passed:" in out
    assert "reason:" in out
    assert "evidence:" in out
    assert "suggestion:" in out
    assert "Response is too casual." in out


def test_inspect_no_judge_verdict_for_non_judge_failure():
    sess, con = _session(["inspect", "skip"])  # default _fail_result = assert_contains
    sess.run()
    assert "judge_verdict:" not in _text(con)


def test_inspect_dict_input():
    case = EvalCase(input={"role": "user", "text": "hello"}, output="hi")
    sess, con = _session(["inspect", "skip"], case=case)
    sess.run()
    out = _text(con)
    assert "role" in out
    assert "user" in out


# ---------------------------------------------------------------------------
# retry command
# ---------------------------------------------------------------------------


def test_retry_pass_shows_pass():
    """retry with an always-passing eval func shows PASS."""
    sess, con = _session(["retry", "skip"], eval_func=lambda c: None)
    sess.run()
    assert "PASS" in _text(con)


def test_retry_still_failing_shows_fail():
    """retry with an always-failing eval func shows FAIL."""
    def _always_fail(case):
        raise EvalAssertionError("still bad", assertion_type="assert_contains")

    sess, con = _session(["retry", "skip"], eval_func=_always_fail)
    sess.run()
    assert "FAIL" in _text(con)
    assert "still failing" in _text(con)


def test_retry_updates_result_to_pass():
    """After a passing retry, the returned result should be passing."""
    sess, _ = _session(["retry", "skip"], eval_func=lambda c: None)
    action, result = sess.run()
    assert result.passed is True


def test_retry_result_still_failing_after_failed_retry():
    """After a failing retry, the returned result should still be failing."""
    def _always_fail(case):
        raise EvalAssertionError("still bad", assertion_type="assert_contains")

    sess, _ = _session(["retry", "skip"], eval_func=_always_fail)
    action, result = sess.run()
    assert result.passed is False


def test_retry_flaky_shows_flaky():
    def _flaky(case):
        raise FlakyEvalError("split verdict")

    sess, con = _session(["retry", "skip"], eval_func=_flaky)
    sess.run()
    assert "FLAKY" in _text(con)


def test_retry_calls_provider_with_new_system_prompt():
    """After hot-swapping system_prompt, retry calls the provider with the new value
    and updates case.output with the fresh response before running the eval."""
    from unittest.mock import AsyncMock, MagicMock

    from sivo.providers import CompletionResult

    provider = MagicMock()
    provider.complete = AsyncMock(
        return_value=CompletionResult(
            output="fresh response",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0001,
            model="claude-haiku-4-5",
        )
    )

    captured: dict = {}

    def _capture(case):
        captured["system_prompt"] = case.system_prompt
        captured["output"] = case.output

    sess, _ = _session(
        ['system_prompt = "Swapped prompt."', "retry", "skip"],
        eval_func=_capture,
        provider=provider,
    )
    sess.run()

    # Provider was called once with the hot-swapped system_prompt
    provider.complete.assert_called_once()
    call_kwargs = provider.complete.call_args.kwargs
    assert call_kwargs["system_prompt"] == "Swapped prompt."

    # case.output was updated with the provider's response before the eval ran
    assert captured.get("output") == "fresh response"


def test_retry_without_hotswap_calls_provider_with_original_system_prompt():
    """retry without any hot-swap still makes a fresh LLM call with the
    original system_prompt and updates case.output."""
    from unittest.mock import AsyncMock, MagicMock

    from sivo.providers import CompletionResult

    provider = MagicMock()
    provider.complete = AsyncMock(
        return_value=CompletionResult(
            output="brand new output",
            input_tokens=8,
            output_tokens=4,
            cost_usd=0.00005,
            model="claude-haiku-4-5",
        )
    )

    case = _case(system_prompt="Original prompt.", output="stale output")
    captured: dict = {}

    def _capture(c):
        captured["output"] = c.output
        captured["system_prompt"] = c.system_prompt

    sess, _ = _session(["retry", "skip"], case=case, eval_func=_capture, provider=provider)
    sess.run()

    # Provider was called with the original system_prompt
    provider.complete.assert_called_once()
    call_kwargs = provider.complete.call_args.kwargs
    assert call_kwargs["system_prompt"] == "Original prompt."

    # case.output was replaced with the provider's response
    assert captured.get("output") == "brand new output"


# ---------------------------------------------------------------------------
# Hot-swap: system_prompt assignment
# ---------------------------------------------------------------------------


def test_set_system_prompt_double_quotes():
    sess, con = _session(['system_prompt = "Be concise."', "skip"])
    sess.run()
    assert "Be concise." in _text(con)
    assert "system_prompt" in _text(con)


def test_set_system_prompt_single_quotes():
    sess, con = _session(["system_prompt = 'Be brief.'", "skip"])
    sess.run()
    assert "Be brief." in _text(con)


def test_set_system_prompt_no_quotes():
    sess, con = _session(["system_prompt = plain value", "skip"])
    sess.run()
    assert "plain value" in _text(con)


def test_set_unknown_field_shows_error():
    sess, con = _session(["temperature = 0.5", "skip"])
    sess.run()
    out = _text(con)
    assert "Cannot set" in out
    assert "temperature" in out


def test_hot_swap_and_retry_uses_new_system_prompt():
    """After setting system_prompt, retry uses the updated value."""
    captured = {}

    def _capture(case):
        captured["system_prompt"] = case.system_prompt
        # Always pass so retry shows PASS
        pass

    sess, _ = _session(
        ['system_prompt = "New prompt."', "retry", "skip"],
        eval_func=_capture,
    )
    sess.run()
    assert captured.get("system_prompt") == "New prompt."


def test_hot_swap_updates_case_for_future_retry():
    """The updated case is retained across multiple retries."""
    calls = []

    def _capture(case):
        calls.append(case.system_prompt)

    sess, _ = _session(
        ['system_prompt = "Prompt A."', "retry", "retry", "skip"],
        eval_func=_capture,
    )
    sess.run()
    # Both retries should see the new system_prompt
    assert all(sp == "Prompt A." for sp in calls)
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# skip preserves original failing result
# ---------------------------------------------------------------------------


def test_skip_returns_original_result():
    result = _fail_result()
    sess, _ = _session(["skip"], result=result)
    action, returned = sess.run()
    assert action == "skip"
    assert returned is result


# ---------------------------------------------------------------------------
# make_pdb_hook factory
# ---------------------------------------------------------------------------


def test_make_pdb_hook_returns_callable():
    con = _console()
    hook = make_pdb_hook(console=con)
    assert callable(hook)


def test_make_pdb_hook_invokes_session():
    """The hook factory should produce a hook that drives a PdbLlmSession."""
    con = _console()
    inputs = iter(["skip"])
    hook = make_pdb_hook(console=con, input_fn=lambda: next(inputs))

    result = _fail_result()
    case = _case()
    action, returned = hook(result, case, lambda c: None)
    assert action == "skip"
    assert returned is result


# ---------------------------------------------------------------------------
# runner integration: pdb_hook parameter
# ---------------------------------------------------------------------------


def test_run_session_pdb_hook_skip(tmp_path):
    """pdb_hook skip → result appended, session continues."""
    from sivo.models import ExecutionRecord
    from sivo.runner import run_session
    from sivo.store import JsonlStore

    store = JsonlStore(tmp_path / ".sivo")
    record = ExecutionRecord(
        id="r1", timestamp="2026-01-01T00:00:00+00:00", run_id="run1",
        input="q", output="bad", model="claude-haiku-4-5",
        input_tokens=1, output_tokens=1, cost_usd=0.0,
    )
    store.write(record)

    eval_file = tmp_path / "eval_test.py"
    eval_file.write_text(
        "from sivo.assertions import EvalAssertionError\n"
        "def eval_check(case):\n"
        "    raise EvalAssertionError('bad', assertion_type='assert_contains')\n"
    )

    hook_calls = []
    def _hook(result, case, eval_func):
        hook_calls.append(result)
        return "skip", result

    session = run_session(
        eval_file, run_id="run1", store=store,
        fail_fast=True, pdb_hook=_hook,
    )
    assert len(hook_calls) == 1
    assert len(session.results) == 1
    assert session.results[0].passed is False


def test_run_session_pdb_hook_abort(tmp_path):
    """pdb_hook abort → session returned immediately."""
    from sivo.models import ExecutionRecord
    from sivo.runner import run_session
    from sivo.store import JsonlStore

    store = JsonlStore(tmp_path / ".sivo")
    for i in range(3):
        store.write(ExecutionRecord(
            id=f"r{i}", timestamp="2026-01-01T00:00:00+00:00", run_id="run1",
            input="q", output="bad", model="claude-haiku-4-5",
            input_tokens=1, output_tokens=1, cost_usd=0.0,
        ))

    eval_file = tmp_path / "eval_test.py"
    eval_file.write_text(
        "from sivo.assertions import EvalAssertionError\n"
        "def eval_check(case):\n"
        "    raise EvalAssertionError('bad', assertion_type='assert_contains')\n"
    )

    hook_calls = []
    def _hook(result, case, eval_func):
        hook_calls.append(result)
        return "abort", result

    session = run_session(
        eval_file, run_id="run1", store=store,
        fail_fast=False, pdb_hook=_hook,
    )
    # Aborted after first failure — only one hook call
    assert len(hook_calls) == 1
    assert len(session.results) == 1


def test_run_session_pdb_hook_continue_disables_hook(tmp_path):
    """pdb_hook continue → hook disabled; remaining failures use fail_fast."""
    from sivo.models import ExecutionRecord
    from sivo.runner import run_session
    from sivo.store import JsonlStore

    store = JsonlStore(tmp_path / ".sivo")
    for i in range(3):
        store.write(ExecutionRecord(
            id=f"r{i}", timestamp="2026-01-01T00:00:00+00:00", run_id="run1",
            input="q", output="bad", model="claude-haiku-4-5",
            input_tokens=1, output_tokens=1, cost_usd=0.0,
        ))

    eval_file = tmp_path / "eval_test.py"
    eval_file.write_text(
        "from sivo.assertions import EvalAssertionError\n"
        "def eval_check(case):\n"
        "    raise EvalAssertionError('bad', assertion_type='assert_contains')\n"
    )

    hook_calls = []
    def _hook(result, case, eval_func):
        hook_calls.append(result)
        return "continue", result  # disables hook, fail_fast re-engages

    session = run_session(
        eval_file, run_id="run1", store=store,
        fail_fast=True, pdb_hook=_hook,
    )
    # Hook called once (first failure), then fail_fast stops on second failure
    assert len(hook_calls) == 1
    assert len(session.results) == 2  # first (hook) + second (fail_fast stop)


def test_run_session_pdb_hook_not_called_on_pass(tmp_path):
    """pdb_hook is not invoked for passing results."""
    from sivo.models import ExecutionRecord
    from sivo.runner import run_session
    from sivo.store import JsonlStore

    store = JsonlStore(tmp_path / ".sivo")
    store.write(ExecutionRecord(
        id="r1", timestamp="2026-01-01T00:00:00+00:00", run_id="run1",
        input="q", output="good", model="claude-haiku-4-5",
        input_tokens=1, output_tokens=1, cost_usd=0.0,
    ))

    eval_file = tmp_path / "eval_test.py"
    eval_file.write_text("def eval_check(case):\n    pass\n")

    hook_calls = []
    def _hook(result, case, eval_func):
        hook_calls.append(result)
        return "skip", result

    run_session(eval_file, run_id="run1", store=store, pdb_hook=_hook)
    assert len(hook_calls) == 0


def test_run_session_pdb_hook_not_called_on_flaky(tmp_path):
    """pdb_hook is not invoked for flaky results."""
    from sivo.assertions import FlakyEvalError
    from sivo.models import ExecutionRecord
    from sivo.runner import run_session
    from sivo.store import JsonlStore

    store = JsonlStore(tmp_path / ".sivo")
    store.write(ExecutionRecord(
        id="r1", timestamp="2026-01-01T00:00:00+00:00", run_id="run1",
        input="q", output="uncertain", model="claude-haiku-4-5",
        input_tokens=1, output_tokens=1, cost_usd=0.0,
    ))

    eval_file = tmp_path / "eval_test.py"
    eval_file.write_text(
        "from sivo.assertions import FlakyEvalError\n"
        "def eval_check(case):\n"
        "    raise FlakyEvalError('split result')\n"
    )

    hook_calls = []
    def _hook(result, case, eval_func):
        hook_calls.append(result)
        return "skip", result

    run_session(eval_file, run_id="run1", store=store, pdb_hook=_hook)
    assert len(hook_calls) == 0
