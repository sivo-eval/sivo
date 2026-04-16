"""Interactive REPL for sivo (--pdb-llm mode).

Pauses execution on a failing eval and drops the developer into an
interactive terminal session where they can inspect context, hot-swap
``system_prompt``, and retry the eval — all without restarting the run.

Commands
--------
inspect          Print ``input``, ``system_prompt``, ``output``, and judge
                 verdict (if the failure was a judge assertion).
retry            Make a fresh LLM call with the current prompt, update
                 ``output``, then re-run the eval assertions against the
                 new response.
skip             Skip this failure and continue to the next eval (REPL
                 remains active for future failures).
continue         Disable the REPL; resume the run without further pausing.
abort            Stop the entire run immediately.

Hot-swap
--------
Edit a field directly at the prompt::

    (pdb-llm) system_prompt = "You are a concise assistant."

Supported settable fields: ``system_prompt``.

After a hot-swap, type ``retry`` to re-run the eval with the new value.
Hot-swapped values are not saved automatically — copy them back to your
source files manually.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Callable

from rich.console import Console
from rich.rule import Rule

if TYPE_CHECKING:
    from sivo.models import EvalCase
    from sivo.providers import Provider
    from sivo.runner import EvalResult

_PROMPT = "(pdb-llm) "
_SETTABLE_FIELDS = {"system_prompt"}
_HELP_LINE = (
    " Commands: inspect  retry  skip  continue  abort"
    '  |  Hot-swap: system_prompt = "new value"  |  retry calls the LLM'
)


class PdbLlmSession:
    """Interactive REPL session for a single failing eval.

    All I/O is injected so the session can be driven headlessly in tests:
    pass a ``Console(file=StringIO())`` and an ``input_fn`` that yields
    scripted command strings.

    Args:
        case:      The :class:`~sivo.models.EvalCase` being evaluated.
        result:    The initial failing :class:`~sivo.runner.EvalResult`.
        eval_func: The eval function to re-run on ``retry``.
        console:   Rich console for all output.
        input_fn:  Callable returning the next command line. Defaults to the
                   built-in ``input()`` (prompt is printed separately so Rich
                   can format it).
    """

    def __init__(
        self,
        case: EvalCase,
        result: EvalResult,
        eval_func: Callable,
        *,
        console: Console,
        input_fn: Callable[[], str] | None = None,
        provider: Provider | None = None,
        model: str = "claude-haiku-4-5",
    ) -> None:
        self._case = case
        self._result = result
        self._eval_func = eval_func
        self._console = console
        self._input_fn = input_fn if input_fn is not None else input
        self._provider = provider
        self._model = model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> tuple[str, EvalResult]:
        """Run the REPL loop until a terminal command is issued.

        Returns:
            ``(action, result)`` where *action* is one of ``"skip"``,
            ``"continue"``, or ``"abort"``, and *result* is the final
            :class:`~sivo.runner.EvalResult` (possibly updated by
            ``retry``).
        """
        self._print_banner()
        while True:
            try:
                self._console.print(_PROMPT, end="")
                line = self._input_fn().strip()
            except (EOFError, KeyboardInterrupt):
                self._console.print()
                return "abort", self._result

            action = self._dispatch(line)
            if action is not None:
                return action, self._result

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, line: str) -> str | None:
        """Dispatch *line* to a handler; return action string or None."""
        if not line:
            return None

        cmd = line.split()[0].lower()

        if cmd == "inspect":
            self._cmd_inspect()
        elif cmd == "retry":
            self._cmd_retry()
        elif cmd == "skip":
            return "skip"
        elif cmd == "continue":
            return "continue"
        elif cmd == "abort":
            return "abort"
        elif "=" in line:
            self._cmd_set(line)
        else:
            self._console.print(f"  Unknown command: {line!r}")
            self._console.print(_HELP_LINE)
        return None

    # ------------------------------------------------------------------
    # Command implementations
    # ------------------------------------------------------------------

    def _cmd_inspect(self) -> None:
        """Print all available fields from the current case and result."""
        con = self._console
        con.print()

        # input — may be str or dict
        inp = self._case.input
        if isinstance(inp, dict):
            con.print("  input:")
            for ln in json.dumps(inp, indent=4).splitlines():
                con.print(f"    {ln}")
        else:
            con.print(f"  input:         {inp}")

        sp = self._case.system_prompt
        con.print(f"  system_prompt: {sp!r}")

        con.print("  output:")
        for ln in str(self._case.output).splitlines():
            con.print(f"    {ln}")

        # Judge verdict (only for assert_judge failures)
        verdict = self._extract_verdict()
        if verdict is not None:
            con.print("  judge_verdict:")
            con.print(f"    passed:     {verdict.passed}")
            con.print(f"    reason:     {verdict.reason}")
            con.print(f"    evidence:   {verdict.evidence}")
            if verdict.suggestion:
                con.print(f"    suggestion: {verdict.suggestion}")
        con.print()

    def _cmd_retry(self) -> None:
        """Make a fresh LLM call then re-run the eval against the new output.

        When a provider is available, calls ``provider.complete()`` with the
        current case state (including any hot-swapped ``system_prompt``),
        updates ``case.output`` with the response, and then runs the eval
        assertions.  If the LLM call fails the error is reported and the REPL
        stays open so the user can retry or abort.

        When no provider is set (headless / test mode), skips the LLM call and
        re-runs assertions against the existing ``case.output``.
        """
        from sivo.runner import EvalEngine

        if self._provider is not None:
            # Build the messages list from the current case state
            if self._case.conversation:
                messages = [
                    {"role": msg.role, "content": msg.content}
                    for msg in self._case.conversation
                ]
            else:
                content = (
                    self._case.input
                    if isinstance(self._case.input, str)
                    else str(self._case.input)
                )
                messages = [{"role": "user", "content": content}]

            try:
                completion = asyncio.run(
                    self._provider.complete(
                        model=self._model,
                        system_prompt=self._case.system_prompt,
                        messages=messages,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                first_line = str(exc).splitlines()[0]
                self._console.print(f"  ERROR  LLM call failed: {first_line}")
                self._console.print(
                    "  (use 'retry' to try again, or 'abort' to stop)"
                )
                return

            self._case = self._case.model_copy(update={"output": completion.output})

        engine = EvalEngine()
        new_result = engine.run(
            self._eval_func,
            self._case,
            eval_name=self._result.eval_name,
            record_id=self._result.record_id,
        )
        self._result = new_result

        if new_result.passed and not new_result.flaky:
            self._console.print("  PASS  (retry succeeded)")
        elif new_result.flaky:
            self._console.print("  FLAKY  (retry inconclusive)")
        else:
            first_line = str(new_result.error).splitlines()[0] if new_result.error else ""
            self._console.print(f"  FAIL  (still failing: {first_line})")

    def _cmd_set(self, line: str) -> None:
        """Handle ``key = value`` field assignment."""
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        # Strip surrounding quotes (single or double)
        if len(value) >= 2 and value[0] in ('"', "'") and value[0] == value[-1]:
            value = value[1:-1]

        if key not in _SETTABLE_FIELDS:
            self._console.print(
                f"  Cannot set {key!r}. Settable fields: "
                + ", ".join(sorted(_SETTABLE_FIELDS))
            )
            return

        self._case = self._case.model_copy(update={key: value})
        self._console.print(f"  {key} = {value!r}  (use 'retry' to re-run)")

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        con = self._console
        con.print()
        con.print(Rule())
        status = "FLAKY" if self._result.flaky else "FAIL"
        con.print(
            f" {status}  {self._result.eval_name}  (record {self._result.record_id})"
        )
        if self._result.error:
            first_line = str(self._result.error).splitlines()[0]
            con.print(f" {first_line}")
        con.print(Rule())
        con.print(" Dropping into pdb-llm. Type 'inspect' to view context.")
        con.print(_HELP_LINE)
        con.print()

    def _extract_verdict(self):
        """Return the JudgeVerdict from the current result, or None."""
        from sivo.assertions import EvalAssertionError
        from sivo.models import JudgeVerdict

        if not isinstance(self._result.error, EvalAssertionError):
            return None
        if self._result.error.assertion_type != "assert_judge":
            return None
        if not isinstance(self._result.error.actual, JudgeVerdict):
            return None
        return self._result.error.actual


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_pdb_hook(
    *,
    console: Console,
    input_fn: Callable[[], str] | None = None,
    provider: Provider | None = None,
    model: str = "claude-haiku-4-5",
) -> Callable:
    """Return a pdb-llm hook for use with :func:`~sivo.runner.run_session`.

    The hook is called for every failing eval result when ``--pdb-llm`` is
    active. It opens a :class:`PdbLlmSession`, interacts with the user, and
    returns ``(action, final_result)``.

    Args:
        console:  Rich console for all REPL output.
        input_fn: Optional injectable input callable (used in tests).
        provider: :class:`~sivo.providers.Provider` instance used to make
                  fresh LLM calls on ``retry``.  When ``None``, ``retry``
                  re-runs assertions against the existing output without
                  calling the LLM (useful in headless / test mode).
        model:    Model string forwarded to ``provider.complete()`` on retry.
                  Defaults to ``"claude-haiku-4-5"``.

    Returns:
        A callable ``(result, case, eval_func) -> (action, result)`` where
        *action* is one of ``"skip"``, ``"continue"``, or ``"abort"``.
    """

    def _hook(
        result: EvalResult,
        case: EvalCase,
        eval_func: Callable,
    ) -> tuple[str, EvalResult]:
        session = PdbLlmSession(
            case=case,
            result=result,
            eval_func=eval_func,
            console=console,
            input_fn=input_fn,
            provider=provider,
            model=model,
        )
        return session.run()

    return _hook
