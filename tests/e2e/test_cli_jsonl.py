"""E2e test: CLI against canned JSONL fixtures.

Verifies that ``sivo run <eval_file> --run-id <id>`` can be invoked as
a subprocess, loads pre-canned ExecutionRecords from JSONL, runs the eval
function against them, and returns the correct exit code.

No LLM calls are made — all outputs come from the JSONL fixture.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared eval file content
# ---------------------------------------------------------------------------

# Passes for any output containing "good"
_EVAL_CONTENT = """\
from sivo.assertions import assert_contains

def eval_check(case):
    assert_contains(case.output, "good")
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(run_id: str, record_id: str, output: str) -> dict:
    return {
        "id": record_id,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "run_id": run_id,
        "input": "test prompt",
        "output": output,
        "model": "claude-haiku-4-5",
        "params": {},
        "input_tokens": 10,
        "output_tokens": 5,
        "cost_usd": 0.0,
        "metadata": {},
        "system_prompt": None,
        "conversation": None,
        "trace": None,
    }


def _setup(tmp_path: Path, run_id: str, outputs: list[str]) -> None:
    """Write the eval file and canned JSONL records under *tmp_path*."""
    (tmp_path / "eval_check.py").write_text(_EVAL_CONTENT)

    records_dir = tmp_path / ".sivo" / "records"
    records_dir.mkdir(parents=True)
    jsonl_path = records_dir / f"{run_id}.jsonl"
    with jsonl_path.open("w") as fh:
        for i, output in enumerate(outputs):
            fh.write(json.dumps(_make_record(run_id, f"rec-{i}", output)) + "\n")


def _run(tmp_path: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "sivo.cli", *extra_args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Scenario A — all pass
# ---------------------------------------------------------------------------


def test_scenario_a_all_pass_exits_zero(tmp_path):
    """All-pass JSONL → exit code 0."""
    run_id = "run_all_pass"
    _setup(tmp_path, run_id, ["good response", "good answer", "good result"])

    result = _run(tmp_path, "run", "eval_check.py", "--run-id", run_id)

    assert result.returncode == 0, (
        f"Expected exit 0 but got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_scenario_a_eval_name_in_output(tmp_path):
    """All-pass run reports the eval function name."""
    run_id = "run_all_pass_name"
    _setup(tmp_path, run_id, ["good response"])

    result = _run(tmp_path, "run", "eval_check.py", "--run-id", run_id)

    combined = result.stdout + result.stderr
    assert "eval_check" in combined


def test_scenario_a_pass_count_in_output(tmp_path):
    """Session summary reports correct passed count."""
    run_id = "run_pass_count"
    _setup(tmp_path, run_id, ["good one", "good two", "good three"])

    result = _run(tmp_path, "run", "eval_check.py", "--run-id", run_id)

    combined = result.stdout + result.stderr
    assert "3 passed" in combined


# ---------------------------------------------------------------------------
# Scenario B — mixed results
# ---------------------------------------------------------------------------


def test_scenario_b_any_fail_exits_one(tmp_path):
    """Mixed JSONL → exit code 1."""
    run_id = "run_mixed"
    _setup(tmp_path, run_id, ["good response", "good answer", "bad response"])

    result = _run(
        tmp_path, "run", "eval_check.py", "--run-id", run_id, "--no-fail-fast"
    )

    assert result.returncode == 1, (
        f"Expected exit 1 but got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_scenario_b_failed_eval_name_in_output(tmp_path):
    """Mixed run reports the failing eval function name."""
    run_id = "run_mixed_name"
    _setup(tmp_path, run_id, ["good response", "bad response"])

    result = _run(
        tmp_path, "run", "eval_check.py", "--run-id", run_id, "--no-fail-fast"
    )

    combined = result.stdout + result.stderr
    assert "eval_check" in combined
    assert "FAIL" in combined


def test_scenario_b_fail_counts_in_output(tmp_path):
    """Session summary reports correct passed/failed counts."""
    run_id = "run_mixed_counts"
    _setup(tmp_path, run_id, ["good one", "good two", "bad three"])

    result = _run(
        tmp_path, "run", "eval_check.py", "--run-id", run_id, "--no-fail-fast"
    )

    combined = result.stdout + result.stderr
    assert "2 passed" in combined
    assert "1 failed" in combined


def test_scenario_b_fail_fast_exits_one(tmp_path):
    """Default fail-fast also exits 1 on first failure."""
    run_id = "run_fail_fast"
    _setup(tmp_path, run_id, ["bad first", "good second", "good third"])

    result = _run(tmp_path, "run", "eval_check.py", "--run-id", run_id)

    assert result.returncode == 1


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_missing_run_id_exits_two(tmp_path):
    """Omitting --run-id (live mode) returns exit code 2 with an error message."""
    (tmp_path / "eval_check.py").write_text(_EVAL_CONTENT)

    result = _run(tmp_path, "run", "eval_check.py")

    assert result.returncode == 2


def test_nonexistent_path_exits_two(tmp_path):
    """A path that does not exist returns exit code 2."""
    result = _run(tmp_path, "run", "nonexistent_dir", "--run-id", "run_x")

    assert result.returncode == 2
