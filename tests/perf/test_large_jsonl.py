"""Performance test: 1000-record JSONL.

Constraints:
- ``sivo replay`` completes in < 30 seconds.
- Peak RSS stays below 200 MB (tracked via tracemalloc).

These tests are marked ``perf`` and skipped by default (they run under CI
only when explicitly selected with ``-m perf``).
"""

from __future__ import annotations

import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import pytest

from sivo.models import ExecutionRecord
from sivo.runner import run_session
from sivo.store import JsonlStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


RECORD_COUNT = 1_000


def _make_store(tmp_path: Path) -> tuple[JsonlStore, str]:
    """Write RECORD_COUNT records under a fresh store; return (store, run_id)."""
    run_id = "perf-run-001"
    store = JsonlStore(tmp_path / ".sivo")

    for i in range(RECORD_COUNT):
        store.write(ExecutionRecord(
            id=f"r{i}",
            timestamp="2026-01-01T00:00:00+00:00",
            run_id=run_id,
            input=f"question {i}",
            output=f"answer {i}",
            model="claude-haiku-4-5",
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.000_01,
        ))

    return store, run_id


def _write_eval_file(tmp_path: Path) -> Path:
    """Write a minimal eval file."""
    eval_file = tmp_path / "eval_perf.py"
    eval_file.write_text(
        "def eval_perf(case):\n"
        "    assert case.output is not None\n"
    )
    return eval_file


# ---------------------------------------------------------------------------
# Wall-clock time test
# ---------------------------------------------------------------------------


@pytest.mark.perf
def test_replay_1000_records_under_30s(tmp_path):
    """Replaying 1000 records through one eval function completes in < 30 s."""
    store, run_id = _make_store(tmp_path)
    eval_file = _write_eval_file(tmp_path)

    start = time.monotonic()
    session = run_session(eval_file, run_id=run_id, store=store, fail_fast=False)
    elapsed = time.monotonic() - start

    assert session.all_passed, f"Some evals failed: {session.failed_count} failures"
    assert len(session.results) == RECORD_COUNT
    assert elapsed < 30, f"Replay took {elapsed:.1f}s (limit: 30s)"


# ---------------------------------------------------------------------------
# Peak memory test
# ---------------------------------------------------------------------------


@pytest.mark.perf
def test_replay_1000_records_memory_under_200mb(tmp_path):
    """Peak heap growth during replay stays below 200 MB."""
    store, run_id = _make_store(tmp_path)
    eval_file = _write_eval_file(tmp_path)

    tracemalloc.start()
    try:
        run_session(eval_file, run_id=run_id, store=store, fail_fast=False)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    peak_mb = peak / 1024 / 1024
    assert peak_mb < 200, f"Peak memory was {peak_mb:.1f} MB (limit: 200 MB)"


# ---------------------------------------------------------------------------
# CLI round-trip wall-clock test
# ---------------------------------------------------------------------------


@pytest.mark.perf
def test_cli_replay_1000_records_under_30s(tmp_path):
    """``sivo replay`` CLI completes 1000 records in < 30 s."""
    store_path = tmp_path / ".sivo"
    _, run_id = _make_store(tmp_path)
    eval_file = _write_eval_file(tmp_path)

    start = time.monotonic()
    result = subprocess.run(
        [
            sys.executable, "-m", "sivo.cli",
            "replay", run_id, str(eval_file),
            "--no-fail-fast",
            "--store-path", str(store_path),
        ],
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - start

    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert elapsed < 30, f"CLI replay took {elapsed:.1f}s (limit: 30s)"
