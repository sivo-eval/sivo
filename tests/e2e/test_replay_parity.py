"""E2e test: live vs replay parity.

Setup
-----
Four canned JSONL records with different outputs trigger a mix of
pass/fail outcomes across two eval functions:

  eval_pass_check  — passes when output contains "good"
  eval_fail_check  — always fails (simulates assertion failure)

Records:
  rec-a: output="good output"   → eval_pass_check: PASS, eval_fail_check: FAIL
  rec-b: output="good output"   → eval_pass_check: PASS, eval_fail_check: FAIL
  rec-c: output="good output"   → eval_pass_check: PASS, eval_fail_check: FAIL
  rec-d: output="good output"   → eval_pass_check: PASS, eval_fail_check: FAIL

With --no-fail-fast we get 4 PASS (eval_pass_check) + 4 FAIL (eval_fail_check).

Scenarios
---------
A  sivo replay <run_id>  --no-fail-fast         → exit 1 (failures present)
B  sivo replay <run_id>  --eval eval_pass_check → exit 0 (only passing eval)
C  sivo replay <run_id>  --filter tag=selected  → only 2 records replayed
D  Anthropic client never instantiated             → zero API calls

Extra metadata records for filter test:
  rec-e: metadata={"tag": "selected"}  → included with --filter tag=selected
  rec-f: metadata={"tag": "other"}     → excluded with --filter tag=selected
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_ID = "run_replay_test"

# ---------------------------------------------------------------------------
# Eval file content
# ---------------------------------------------------------------------------

_EVAL_CONTENT = """\
from sivo.assertions import EvalAssertionError, assert_contains


def eval_pass_check(case):
    assert_contains(case.output, "good")


def eval_fail_check(case):
    raise EvalAssertionError(
        "eval_fail_check always fails",
        assertion_type="assert_contains",
    )
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(record_id: str, output: str = "good output", metadata: dict | None = None) -> dict:
    return {
        "id": record_id,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "run_id": RUN_ID,
        "input": "test prompt",
        "output": output,
        "model": "claude-haiku-4-5",
        "params": {},
        "input_tokens": 10,
        "output_tokens": 5,
        "cost_usd": 0.0001,
        "metadata": metadata or {},
        "system_prompt": None,
        "conversation": None,
        "trace": None,
    }


def _build_project(tmp_path: Path) -> None:
    (tmp_path / "eval_replay.py").write_text(_EVAL_CONTENT)

    records_dir = tmp_path / ".sivo" / "records"
    records_dir.mkdir(parents=True)

    records = [
        _make_record("rec-a"),
        _make_record("rec-b"),
        _make_record("rec-c"),
        _make_record("rec-d"),
    ]
    lines = "\n".join(json.dumps(r) for r in records) + "\n"
    (records_dir / f"{RUN_ID}.jsonl").write_text(lines)


def _build_project_with_filter_records(tmp_path: Path) -> None:
    """Build a project with two metadata-tagged records for filter testing."""
    (tmp_path / "eval_replay.py").write_text(_EVAL_CONTENT)

    records_dir = tmp_path / ".sivo" / "records"
    records_dir.mkdir(parents=True)

    records = [
        _make_record("rec-e", metadata={"tag": "selected"}),
        _make_record("rec-f", metadata={"tag": "other"}),
    ]
    lines = "\n".join(json.dumps(r) for r in records) + "\n"
    (records_dir / f"{RUN_ID}.jsonl").write_text(lines)


def _run(tmp_path: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "sivo.cli", "replay", RUN_ID,
         "eval_replay.py", "--no-fail-fast", *extra_args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def _run_minimal(tmp_path: Path, *extra_args: str) -> subprocess.CompletedProcess:
    """Run replay without --no-fail-fast (uses fail_fast default)."""
    return subprocess.run(
        [sys.executable, "-m", "sivo.cli", "replay", RUN_ID,
         "eval_replay.py", *extra_args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def _out(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# Scenario A: default run with failures
# ---------------------------------------------------------------------------


def test_replay_exit_one_on_failures(tmp_path):
    """With failures present, exit code should be 1."""
    _build_project(tmp_path)
    proc = _run(tmp_path)
    assert proc.returncode == 1, _out(proc)


def test_replay_shows_pass_label(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path)
    assert "PASS" in _out(proc)


def test_replay_shows_fail_label(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path)
    assert "FAIL" in _out(proc)


def test_replay_shows_pass_count(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path)
    assert "4 passed" in _out(proc)


def test_replay_shows_fail_count(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path)
    assert "4 failed" in _out(proc)


def test_replay_shows_run_id(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path)
    assert RUN_ID in _out(proc)


# ---------------------------------------------------------------------------
# Scenario B: --eval filter
# ---------------------------------------------------------------------------


def test_replay_eval_filter_exit_zero(tmp_path):
    """--eval eval_pass_check should exit 0 (only the passing eval runs)."""
    _build_project(tmp_path)
    proc = _run(tmp_path, "--eval", "eval_pass_check")
    assert proc.returncode == 0, _out(proc)


def test_replay_eval_filter_shows_only_that_eval(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "--eval", "eval_pass_check")
    out = _out(proc)
    assert "eval_pass_check" in out
    assert "eval_fail_check" not in out


def test_replay_eval_filter_pass_count(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "--eval", "eval_pass_check")
    assert "4 passed" in _out(proc)


# ---------------------------------------------------------------------------
# Scenario C: --filter metadata filter
# ---------------------------------------------------------------------------


def test_replay_metadata_filter_restricts_records(tmp_path):
    """--filter tag=selected should replay only the matching record."""
    _build_project_with_filter_records(tmp_path)
    proc = _run(tmp_path, "--filter", "tag=selected")
    out = _out(proc)
    # Only 1 record passes eval_pass_check (rec-e), 1 fails eval_fail_check
    assert "1 passed" in out
    assert "1 failed" in out


def test_replay_metadata_filter_exit_one(tmp_path):
    _build_project_with_filter_records(tmp_path)
    proc = _run(tmp_path, "--filter", "tag=selected")
    assert proc.returncode == 1, _out(proc)


# ---------------------------------------------------------------------------
# Scenario D: no LLM calls
# ---------------------------------------------------------------------------


def test_replay_no_lm_calls(tmp_path):
    """Verify no Anthropic API calls are made during replay.

    We do this by checking there's no ImportError or API key error in output
    and the process exits without importing anthropic in a way that calls it.
    The subprocess approach means we can't mock directly, so we verify the
    behaviour indirectly: replay works correctly without ANTHROPIC_API_KEY set.
    """
    _build_project(tmp_path)
    import os
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    proc = subprocess.run(
        [sys.executable, "-m", "sivo.cli", "replay", RUN_ID,
         "eval_replay.py", "--no-fail-fast"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    # If any LLM call was attempted, we'd get an auth error; instead we get
    # a clean result (exit 1 only because eval_fail_check always fails)
    assert proc.returncode == 1, _out(proc)
    assert "error" not in proc.stderr.lower() or "sivo: error:" not in proc.stderr
    assert RUN_ID in _out(proc)
