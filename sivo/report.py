"""Terminal UI and session receipt for sivo.

Uses Rich for formatting. All public functions accept an optional ``console``
parameter so callers can redirect or capture output in tests.

Verbosity levels
----------------
0 (default)  Pass/fail/flaky per result + one-sentence reason on failure.
1 (-v)       Expanded: for judge failures, show reason, evidence, suggestion.
             For other failures, show the full error message.
2 (-vv)      Debug: for judge failures, show full JudgeVerdict as JSON.
             Show all error context for every failure.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.rule import Rule

if TYPE_CHECKING:
    from sivo.runner import EvalResult, SessionResult

# ---------------------------------------------------------------------------
# Console factory
# ---------------------------------------------------------------------------

_INDENT = "       "  # 7-space indent for detail lines under a result


def make_console(*, stderr: bool = False) -> Console:
    """Create a Rich console suitable for sivo output.

    Disables markup highlighting (prevents accidental rich-markup
    interpretation of eval output text) while keeping colour when the
    output is a terminal.
    """
    import sys

    return Console(
        file=sys.stderr if stderr else sys.stdout,
        highlight=False,
        markup=False,
    )


# ---------------------------------------------------------------------------
# Per-result formatting
# ---------------------------------------------------------------------------


def _extract_judge_verdict(result: EvalResult):
    """Return the JudgeVerdict from a judge assertion failure, or None."""
    from sivo.assertions import EvalAssertionError
    from sivo.models import JudgeVerdict

    if not isinstance(result.error, EvalAssertionError):
        return None
    if result.error.assertion_type != "assert_judge":
        return None
    if not isinstance(result.error.actual, JudgeVerdict):
        return None
    return result.error.actual


def _status_label(result: EvalResult) -> str:
    if result.flaky:
        return "FLAKY"
    return "PASS" if result.passed else "FAIL"


def print_result(
    result: EvalResult,
    *,
    verbose: int = 0,
    console: Console | None = None,
) -> None:
    """Print a single eval result with optional detail lines.

    Args:
        result:  The :class:`~sivo.runner.EvalResult` to render.
        verbose: Verbosity level (0, 1, or 2).
        console: Rich console to write to. A default stdout console is
                 created if not provided.
    """
    con = console or make_console()
    label = _status_label(result)
    con.print(f"{label}  {result.eval_name} (record {result.record_id})")

    if result.passed and not result.flaky:
        return  # nothing more to print for clean passes

    if result.error is None:
        return

    verdict = _extract_judge_verdict(result)

    if verbose >= 2 and verdict is not None:
        # Full JudgeVerdict JSON
        verdict_json = json.dumps(verdict.model_dump(), indent=2)
        for line in verdict_json.splitlines():
            con.print(f"{_INDENT}{line}")

    elif verbose >= 1 and verdict is not None:
        # Structured judge fields
        con.print(f"{_INDENT}Reason:     {verdict.reason}")
        con.print(f"{_INDENT}Evidence:   {verdict.evidence}")
        if verdict.suggestion:
            con.print(f"{_INDENT}Suggestion: {verdict.suggestion}")

    elif verbose >= 1:
        # Full error text for non-judge failures
        for line in str(result.error).splitlines():
            con.print(f"{_INDENT}{line}")

    else:
        # Default: first line of the error message
        first_line = str(result.error).splitlines()[0] if result.error else ""
        if first_line:
            con.print(f"{_INDENT}{first_line}")


# ---------------------------------------------------------------------------
# Session receipt
# ---------------------------------------------------------------------------


def print_receipt(
    session: SessionResult,
    *,
    console: Console | None = None,
) -> None:
    """Print the boxed session receipt.

    Displays counts, token totals, cost, and run id. Always printed after
    per-result lines regardless of verbosity.

    Args:
        session: The completed :class:`~sivo.runner.SessionResult`.
        console: Rich console to write to.
    """
    con = console or make_console()

    passed = session.passed_count
    failed = session.failed_count
    flaky = session.flaky_count
    total = passed + failed + flaky

    # Build count string
    counts = f"{passed} passed"
    if failed:
        counts += f"  {failed} failed"
    if flaky:
        counts += f"  {flaky} flaky"

    # Cost breakdown — show per-eval only when more than one eval ran
    cost_str = f"${session.total_cost_usd:.6f}"
    if len(session.cost_by_eval) > 1:
        breakdown = "  ".join(
            f"{name}: ${cost:.6f}"
            for name, cost in sorted(session.cost_by_eval.items())
            if cost > 0
        )
        if breakdown:
            cost_str += f"  ({breakdown})"

    con.print()
    con.print(Rule())
    con.print(" sivo run complete")
    con.print(f" {counts}")
    if total > 0:
        con.print(
            f" tokens: {session.total_input_tokens:,} in"
            f" / {session.total_output_tokens:,} out"
        )
    con.print(f" cost:   {cost_str}")
    con.print(f" run id: {session.run_id}")
    con.print(Rule())


# ---------------------------------------------------------------------------
# Full session output (per-result + receipt)
# ---------------------------------------------------------------------------


def print_session(
    session: SessionResult,
    *,
    verbose: int = 0,
    console: Console | None = None,
) -> None:
    """Print all results followed by the session receipt.

    This is a convenience wrapper used by the CLI when all results are
    available at once. For streaming output (via the ``on_result`` callback)
    use :func:`print_result` and :func:`print_receipt` directly.

    Args:
        session: The completed session.
        verbose: Verbosity level passed to :func:`print_result`.
        console: Rich console to write to.
    """
    con = console or make_console()
    for result in session.results:
        print_result(result, verbose=verbose, console=con)
    print_receipt(session, console=con)


# ---------------------------------------------------------------------------
# JUnit XML
# ---------------------------------------------------------------------------


def write_junit_xml(
    session: SessionResult,
    path: Path | str,
    *,
    strict_flaky: bool = False,
) -> None:
    """Write a JUnit-compatible XML file for *session*.

    The XML format is accepted natively by GitHub Actions, Jenkins,
    CircleCI, and GitLab CI without any glue code.

    Structure::

        <testsuites>
          <testsuite name="sivo" tests=N failures=M skipped=K errors=0>
            <testcase name="eval_tone[rec-1]" classname="run_id" time="0.0"/>
            <testcase name="eval_check[rec-2]" classname="run_id" time="0.0">
              <failure type="EvalAssertionError" message="...">full text</failure>
            </testcase>
            <testcase name="eval_flaky[rec-3]" classname="run_id" time="0.0">
              <skipped message="FLAKY: ..."/>
            </testcase>
          </testsuite>
        </testsuites>

    Args:
        session:      Completed session to serialise.
        path:         File path to write. Parent directories are created.
        strict_flaky: When ``True``, flaky results are written as
                      ``<failure>`` elements instead of ``<skipped>``.
    """
    failures = session.failed_count + (session.flaky_count if strict_flaky else 0)
    skipped = 0 if strict_flaky else session.flaky_count
    total = len(session.results)

    suite = ET.Element(
        "testsuite",
        attrib={
            "name": "sivo",
            "tests": str(total),
            "failures": str(failures),
            "skipped": str(skipped),
            "errors": "0",
            "time": "0.0",
        },
    )

    for result in session.results:
        tc = ET.SubElement(
            suite,
            "testcase",
            attrib={
                "name": f"{result.eval_name}[{result.record_id}]",
                "classname": session.run_id,
                "time": "0.0",
            },
        )

        if result.flaky and not strict_flaky:
            # Flaky → skipped (does not fail CI by default)
            msg = str(result.error).splitlines()[0] if result.error else "FLAKY"
            ET.SubElement(tc, "skipped", attrib={"message": f"FLAKY: {msg}"})
        elif not result.passed or (result.flaky and strict_flaky):
            # Hard failure or strict-flaky flaky → failure
            first_line = str(result.error).splitlines()[0] if result.error else "FAIL"
            err_type = type(result.error).__name__ if result.error else "AssertionError"
            full_text = str(result.error) if result.error else ""
            failure = ET.SubElement(
                tc,
                "failure",
                attrib={"type": err_type, "message": first_line},
            )
            failure.text = full_text

    root = ET.Element("testsuites")
    root.append(suite)
    tree = ET.ElementTree(root)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out), encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# Cost warning
# ---------------------------------------------------------------------------


def print_cost_warning(
    session: SessionResult,
    warn_above_usd: float,
    *,
    console: Console | None = None,
) -> None:
    """Print a cost warning when session cost exceeds *warn_above_usd*.

    Designed to be called after :func:`print_receipt` when a cost threshold
    is configured in ``sivo.toml``.

    Args:
        session:         Completed session.
        warn_above_usd:  Threshold from ``[sivo.cost] warn_above_usd``.
        console:         Rich console to write to.
    """
    if session.total_cost_usd <= warn_above_usd:
        return
    con = console or make_console()
    con.print(
        f"[yellow]WARNING:[/yellow] session cost ${session.total_cost_usd:.6f} "
        f"exceeded warn_above_usd=${warn_above_usd:.2f} "
        "(set in sivo.toml [sivo.cost])",
        markup=True,
    )


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------


def write_json_summary(session: SessionResult, store_path: Path | str) -> None:
    """Write a JSON summary of *session* to ``<store_path>/results/<run_id>.json``.

    Called automatically after every ``sivo run`` and ``sivo replay``
    invocation so that CI pipelines can consume structured results without
    parsing terminal output.

    Args:
        session:    Completed session to serialise.
        store_path: Root of the sivo data store (typically ``.sivo``).
    """
    results_dir = Path(store_path) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "run_id": session.run_id,
        "passed": session.passed_count,
        "failed": session.failed_count,
        "flaky": session.flaky_count,
        "total": len(session.results),
        "total_cost_usd": session.total_cost_usd,
        "total_input_tokens": session.total_input_tokens,
        "total_output_tokens": session.total_output_tokens,
        "results": [
            {
                "eval_name": r.eval_name,
                "record_id": r.record_id,
                "passed": r.passed,
                "flaky": r.flaky,
                "error": str(r.error) if r.error else None,
            }
            for r in session.results
        ],
    }

    out = results_dir / f"{session.run_id}.json"
    out.write_text(json.dumps(data, indent=2))
