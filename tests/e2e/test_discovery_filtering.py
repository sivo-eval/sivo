"""E2e test: eval discovery and --eval filtering.

Mini project layout under tmp_path:
    evals/
        eval_tone.py          — def eval_tone(case)
        eval_accuracy.py      — def eval_accuracy(case)
        sub/
            eval_edge_cases.py — def eval_edge_cases(case)
    .sivo/records/
        run_discover.jsonl    — 1 record whose output satisfies all three evals

Scenarios
---------
A  sivo run evals/ --run-id ...        → 3 eval functions run, exit 0
B  sivo run evals/eval_tone.py ...     → 1 eval function run, exit 0
C  sivo run evals/ --eval eval_edge_cases ... → 1 eval function, exit 0
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project template
# ---------------------------------------------------------------------------

# A single output that satisfies all three eval assertions.
_PASSING_OUTPUT = "correct tone and accurate edge case"

_EVAL_TONE = """\
from sivo.assertions import assert_contains

def eval_tone(case):
    assert_contains(case.output, "tone")
"""

_EVAL_ACCURACY = """\
from sivo.assertions import assert_contains

def eval_accuracy(case):
    assert_contains(case.output, "accurate")
"""

_EVAL_EDGE_CASES = """\
from sivo.assertions import assert_contains

def eval_edge_cases(case):
    assert_contains(case.output, "edge case")
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RUN_ID = "run_discover"


def _build_project(tmp_path: Path) -> None:
    """Write the mini eval project and the canned JSONL record."""
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "eval_tone.py").write_text(_EVAL_TONE)
    (evals / "eval_accuracy.py").write_text(_EVAL_ACCURACY)
    sub = evals / "sub"
    sub.mkdir()
    (sub / "eval_edge_cases.py").write_text(_EVAL_EDGE_CASES)

    records_dir = tmp_path / ".sivo" / "records"
    records_dir.mkdir(parents=True)
    record = {
        "id": "rec-0",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "run_id": RUN_ID,
        "input": "test prompt",
        "output": _PASSING_OUTPUT,
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
    (records_dir / f"{RUN_ID}.jsonl").write_text(json.dumps(record) + "\n")


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "sivo.cli", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def _output(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# Scenario A — run full directory
# ---------------------------------------------------------------------------


def test_scenario_a_exits_zero(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "run", "evals/", "--run-id", RUN_ID)
    assert proc.returncode == 0, _output(proc)


def test_scenario_a_all_three_eval_names_in_output(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "run", "evals/", "--run-id", RUN_ID)
    out = _output(proc)
    assert "eval_tone" in out
    assert "eval_accuracy" in out
    assert "eval_edge_cases" in out


def test_scenario_a_count_of_three_in_output(tmp_path):
    """Three PASS lines appear — one per discovered eval function."""
    _build_project(tmp_path)
    proc = _run(tmp_path, "run", "evals/", "--run-id", RUN_ID)
    out = _output(proc)
    assert out.count("PASS") == 3, f"Expected 3 PASS lines, got:\n{out}"


def test_scenario_a_summary_shows_three_passed(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "run", "evals/", "--run-id", RUN_ID)
    assert "3 passed" in _output(proc)


# ---------------------------------------------------------------------------
# Scenario B — single file
# ---------------------------------------------------------------------------


def test_scenario_b_exits_zero(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "run", "evals/eval_tone.py", "--run-id", RUN_ID)
    assert proc.returncode == 0, _output(proc)


def test_scenario_b_only_one_eval_in_output(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "run", "evals/eval_tone.py", "--run-id", RUN_ID)
    out = _output(proc)
    assert out.count("PASS") == 1, f"Expected 1 PASS line, got:\n{out}"


def test_scenario_b_correct_eval_name_in_output(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "run", "evals/eval_tone.py", "--run-id", RUN_ID)
    out = _output(proc)
    assert "eval_tone" in out
    assert "eval_accuracy" not in out
    assert "eval_edge_cases" not in out


def test_scenario_b_summary_shows_one_passed(tmp_path):
    _build_project(tmp_path)
    proc = _run(tmp_path, "run", "evals/eval_tone.py", "--run-id", RUN_ID)
    assert "1 passed" in _output(proc)


# ---------------------------------------------------------------------------
# Scenario C — --eval filter across directory depth
# ---------------------------------------------------------------------------


def test_scenario_c_exits_zero(tmp_path):
    _build_project(tmp_path)
    proc = _run(
        tmp_path, "run", "evals/", "--run-id", RUN_ID, "--eval", "eval_edge_cases"
    )
    assert proc.returncode == 0, _output(proc)


def test_scenario_c_only_one_eval_in_output(tmp_path):
    _build_project(tmp_path)
    proc = _run(
        tmp_path, "run", "evals/", "--run-id", RUN_ID, "--eval", "eval_edge_cases"
    )
    out = _output(proc)
    assert out.count("PASS") == 1, f"Expected 1 PASS line, got:\n{out}"


def test_scenario_c_correct_eval_name_in_output(tmp_path):
    _build_project(tmp_path)
    proc = _run(
        tmp_path, "run", "evals/", "--run-id", RUN_ID, "--eval", "eval_edge_cases"
    )
    out = _output(proc)
    assert "eval_edge_cases" in out
    assert "eval_tone" not in out
    assert "eval_accuracy" not in out


def test_scenario_c_filter_finds_nested_eval(tmp_path):
    """--eval matches eval_edge_cases even though it lives in evals/sub/."""
    _build_project(tmp_path)
    proc = _run(
        tmp_path, "run", "evals/", "--run-id", RUN_ID, "--eval", "eval_edge_cases"
    )
    assert proc.returncode == 0, _output(proc)


def test_scenario_c_summary_shows_one_passed(tmp_path):
    _build_project(tmp_path)
    proc = _run(
        tmp_path, "run", "evals/", "--run-id", RUN_ID, "--eval", "eval_edge_cases"
    )
    assert "1 passed" in _output(proc)
