"""Unit tests for sivo.store (JsonlStore)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sivo.models import ExecutionRecord
from sivo.store import JsonlStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(**overrides) -> ExecutionRecord:
    defaults = dict(
        id="rec-1",
        timestamp="2026-01-01T00:00:00+00:00",
        run_id="run_test",
        input="Hello",
        output="World",
        model="claude-haiku-4-5",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0,
    )
    defaults.update(overrides)
    return ExecutionRecord(**defaults)


# ---------------------------------------------------------------------------
# write / read round-trip
# ---------------------------------------------------------------------------


def test_write_creates_file(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    rec = _record()
    store.write(rec)
    assert store._records_file("run_test").exists()


def test_write_read_round_trip(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    rec = _record()
    store.write(rec)

    loaded = store.read("run_test")
    assert len(loaded) == 1
    assert loaded[0].id == rec.id
    assert loaded[0].output == rec.output


def test_write_multiple_records_same_run(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    r1 = _record(id="rec-1", input="A", output="AA")
    r2 = _record(id="rec-2", input="B", output="BB")

    store.write(r1)
    store.write(r2)

    loaded = store.read("run_test")
    assert len(loaded) == 2
    ids = {r.id for r in loaded}
    assert ids == {"rec-1", "rec-2"}


def test_write_different_runs_different_files(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    r1 = _record(id="rec-1", run_id="run_a")
    r2 = _record(id="rec-2", run_id="run_b")

    store.write(r1)
    store.write(r2)

    assert store.read("run_a") == [r1]
    assert store.read("run_b") == [r2]


def test_read_nonexistent_run_returns_empty(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    assert store.read("no_such_run") == []


def test_write_is_append_only(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    r1 = _record(id="rec-1")
    r2 = _record(id="rec-2")

    store.write(r1)
    store.write(r2)

    # Verify the JSONL file has two lines
    path = store._records_file("run_test")
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2


def test_written_lines_are_valid_json(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(id="rec-1"))
    store.write(_record(id="rec-2"))

    path = store._records_file("run_test")
    for line in path.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            assert "id" in obj
            assert "run_id" in obj


def test_read_preserves_all_fields(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    rec = _record(
        id="rec-full",
        system_prompt="You are helpful.",
        metadata={"tag": "smoke", "dataset": "v1"},
        input_tokens=42,
        output_tokens=7,
        cost_usd=0.000123,
    )
    store.write(rec)
    [loaded] = store.read("run_test")

    assert loaded.system_prompt == "You are helpful."
    assert loaded.metadata == {"tag": "smoke", "dataset": "v1"}
    assert loaded.input_tokens == 42
    assert loaded.output_tokens == 7
    assert loaded.cost_usd == pytest.approx(0.000123)


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


def test_list_runs_empty_store(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    assert store.list_runs() == []


def test_list_runs_single(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(run_id="run_alpha"))
    assert store.list_runs() == ["run_alpha"]


def test_list_runs_multiple_sorted(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    for run_id in ("run_c", "run_a", "run_b"):
        store.write(_record(run_id=run_id, id=run_id))
    assert store.list_runs() == ["run_a", "run_b", "run_c"]


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def test_filter_no_filters_returns_all(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(id="r1", metadata={"x": 1}))
    store.write(_record(id="r2", metadata={"x": 2}))
    assert len(store.filter("run_test")) == 2


def test_filter_by_metadata_key(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(id="r1", metadata={"model": "haiku", "tag": "fast"}))
    store.write(_record(id="r2", metadata={"model": "sonnet", "tag": "fast"}))
    store.write(_record(id="r3", metadata={"model": "haiku", "tag": "slow"}))

    result = store.filter("run_test", model="haiku")
    assert {r.id for r in result} == {"r1", "r3"}


def test_filter_multiple_conditions(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(id="r1", metadata={"model": "haiku", "tag": "fast"}))
    store.write(_record(id="r2", metadata={"model": "haiku", "tag": "slow"}))
    store.write(_record(id="r3", metadata={"model": "sonnet", "tag": "fast"}))

    result = store.filter("run_test", model="haiku", tag="fast")
    assert len(result) == 1
    assert result[0].id == "r1"


def test_filter_no_matches_returns_empty(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    store.write(_record(id="r1", metadata={"model": "haiku"}))
    assert store.filter("run_test", model="opus") == []


def test_filter_nonexistent_run_returns_empty(tmp_path):
    store = JsonlStore(tmp_path / ".sivo")
    assert store.filter("no_such_run", model="haiku") == []
