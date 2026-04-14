"""E2e test: output format correctness across verbosity levels.

Setup
-----
One canned JSONL record whose output triggers all three outcome types when
processed by three eval functions in a single eval file:

  eval_pass_check  — always passes          → PASS
  eval_tone_check  — simulates judge FAIL    → FAIL  (with JudgeVerdict evidence)
  eval_flaky_check — simulates FLAKY result  → FLAKY

This gives exactly 1 PASS, 1 FAIL, 1 FLAKY from a single invocation.

Scenarios
---------
default  verify pass/fail/flaky counts and run id in output
-v       verify failure evidence text appears
-vv      verify full JudgeVerdict JSON fields appear
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_ID = "run_fmt_test"

# Output text that allows all three eval functions to demonstrate their
# respective outcomes from a single record.
_OUTPUT = "wrong tone and uncertain result"

# ---------------------------------------------------------------------------
# Eval file content
# ---------------------------------------------------------------------------

_EVAL_CONTENT = """\
from sivo.assertions import EvalAssertionError, FlakyEvalError
from sivo.models import JudgeVerdict


def eval_pass_check(case):
    # Always passes
    pass


def eval_tone_check(case):
    # Simulates a judge failure with full JudgeVerdict evidence
    if "wrong" in case.output:
        verdict = JudgeVerdict(
            passed=False,
            reason="The response uses an inappropriate tone.",
            evidence='"wrong tone and uncertain result"',
            suggestion="Use more respectful language.",
        )
        raise EvalAssertionError(
            "Judge assertion failed (rubric='tone').\\n"
            f"Reason: {verdict.reason}\\n"
            f"Evidence: {verdict.evidence!r}",
            assertion_type="assert_judge",
            expected="pass rubric 'tone'",
            actual=verdict,
        )


def eval_flaky_check(case):
    # Simulates a flaky (indeterminate) result
    if "uncertain" in case.output:
        raise FlakyEvalError("Split judge results (1 pass, 2 fail across 3 attempts).")
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_project(tmp_path: Path) -> None:
    (tmp_path / "eval_formats.py").write_text(_EVAL_CONTENT)

    records_dir = tmp_path / ".sivo" / "records"
    records_dir.mkdir(parents=True)
    record = {
        "id": "rec-0",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "run_id": RUN_ID,
        "input": "test prompt",
        "output": _OUTPUT,
        "model": "claude-haiku-4-5",
        "params": {},
        "input_tokens": 15,
        "output_tokens": 8,
        "cost_usd": 0.000044,
        "metadata": {},
        "system_prompt": None,
        "conversation": None,
        "trace": None,
    }
    (records_dir / f"{RUN_ID}.jsonl").write_text(json.dumps(record) + "\n")


def _run(tmp_path: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "sivo.cli", "run", "eval_formats.py",
         "--run-id", RUN_ID, "--no-fail-fast", *extra_args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def _out(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# Default verbosity
# ---------------------------------------------------------------------------


def test_default_exit_one_due_to_fail(tmp_path):
    """A FAIL result causes exit code 1 even when FLAKY is present."""
    _build_project(tmp_path)
    proc = _run(tmp_path)
    assert proc.returncode == 1, _out(proc)


def test_default_contains_pass_count(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path)
    assert "1 passed" in _out(proc)


def test_default_contains_fail_count(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path)
    assert "1 failed" in _out(proc)


def test_default_contains_flaky_count(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path)
    assert "1 flaky" in _out(proc)


def test_default_contains_run_id(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path)
    assert RUN_ID in _out(proc)


def test_default_pass_label_present(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path)
    assert "PASS" in _out(proc)


def test_default_fail_label_present(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path)
    assert "FAIL" in _out(proc)


def test_default_flaky_label_present(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path)
    assert "FLAKY" in _out(proc)


def test_default_no_evidence_shown(tmp_path):
    """At default verbosity, Evidence: field is NOT shown."""
    _build_project(tmp_path)
    proc = _run(tmp_path)
    assert "Evidence:" not in _out(proc)


# ---------------------------------------------------------------------------
# -v verbosity
# ---------------------------------------------------------------------------


def test_v_shows_evidence(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "-v")
    assert "Evidence:" in _out(proc)


def test_v_shows_evidence_text(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "-v")
    # The evidence field from the JudgeVerdict in the eval file
    assert "wrong tone and uncertain result" in _out(proc)


def test_v_shows_reason(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "-v")
    assert "Reason:" in _out(proc)
    assert "inappropriate tone" in _out(proc)


def test_v_shows_suggestion(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "-v")
    assert "Suggestion:" in _out(proc)


def test_v_still_shows_counts(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "-v")
    out = _out(proc)
    assert "1 passed" in out
    assert "1 failed" in out
    assert "1 flaky" in out


# ---------------------------------------------------------------------------
# -vv verbosity
# ---------------------------------------------------------------------------


def test_vv_shows_json_passed_key(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "-vv")
    assert '"passed"' in _out(proc)


def test_vv_shows_json_reason_key(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "-vv")
    assert '"reason"' in _out(proc)


def test_vv_shows_json_evidence_key(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "-vv")
    assert '"evidence"' in _out(proc)


def test_vv_shows_json_suggestion_key(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "-vv")
    assert '"suggestion"' in _out(proc)


def test_vv_json_is_parseable(tmp_path):
    """The JSON block in -vv output should be valid JSON."""
    _build_project(tmp_path)
    proc = _run(tmp_path, "-vv")
    out = _out(proc)
    # Find the JSON block (first { to matching })
    start = out.index("{")
    # Walk to find balanced closing brace
    depth = 0
    end = start
    for i, ch in enumerate(out[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    parsed = json.loads(out[start:end])
    assert parsed["passed"] is False
    assert "reason" in parsed


def test_vv_still_shows_counts(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "-vv")
    out = _out(proc)
    assert "1 passed" in out
    assert "1 failed" in out
