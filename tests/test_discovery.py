"""Unit tests for sivo.discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from sivo.discovery import (
    DiscoveredEval,
    discover,
    discover_eval_files,
    load_eval_functions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# discover_eval_files
# ---------------------------------------------------------------------------


def test_discover_eval_files_single_file(tmp_path):
    f = _write(tmp_path / "eval_tone.py", "")
    assert discover_eval_files(f) == [f]


def test_discover_eval_files_non_eval_file_returns_empty(tmp_path):
    f = _write(tmp_path / "helpers.py", "")
    assert discover_eval_files(f) == []


def test_discover_eval_files_directory(tmp_path):
    _write(tmp_path / "eval_a.py", "")
    _write(tmp_path / "eval_b.py", "")
    _write(tmp_path / "helpers.py", "")

    result = discover_eval_files(tmp_path)
    names = [p.name for p in result]
    assert sorted(names) == ["eval_a.py", "eval_b.py"]


def test_discover_eval_files_recursive(tmp_path):
    _write(tmp_path / "eval_top.py", "")
    _write(tmp_path / "sub" / "eval_nested.py", "")
    _write(tmp_path / "sub" / "deep" / "eval_deep.py", "")

    result = discover_eval_files(tmp_path)
    names = sorted(p.name for p in result)
    assert names == ["eval_deep.py", "eval_nested.py", "eval_top.py"]


def test_discover_eval_files_empty_directory(tmp_path):
    assert discover_eval_files(tmp_path) == []


def test_discover_eval_files_sorted(tmp_path):
    _write(tmp_path / "eval_z.py", "")
    _write(tmp_path / "eval_a.py", "")
    _write(tmp_path / "eval_m.py", "")
    result = discover_eval_files(tmp_path)
    assert result == sorted(result)


# ---------------------------------------------------------------------------
# load_eval_functions
# ---------------------------------------------------------------------------


def test_load_eval_functions_finds_eval_func(tmp_path):
    f = _write(
        tmp_path / "eval_tone.py",
        "def eval_tone(case): pass\n",
    )
    evals = load_eval_functions(f)
    assert len(evals) == 1
    assert evals[0].name == "eval_tone"
    assert callable(evals[0].func)
    assert evals[0].source_file == f


def test_load_eval_functions_ignores_non_eval_names(tmp_path):
    f = _write(
        tmp_path / "eval_misc.py",
        "def eval_tone(case): pass\ndef helper(): pass\nclass Foo: pass\n",
    )
    evals = load_eval_functions(f)
    assert len(evals) == 1
    assert evals[0].name == "eval_tone"


def test_load_eval_functions_multiple_funcs(tmp_path):
    f = _write(
        tmp_path / "eval_multi.py",
        "def eval_b(case): pass\ndef eval_a(case): pass\n",
    )
    evals = load_eval_functions(f)
    names = [e.name for e in evals]
    assert names == ["eval_a", "eval_b"]  # sorted


def test_load_eval_functions_no_eval_funcs(tmp_path):
    f = _write(tmp_path / "eval_empty.py", "x = 1\n")
    assert load_eval_functions(f) == []


def test_load_eval_functions_detects_cases_func(tmp_path):
    f = _write(
        tmp_path / "eval_tone.py",
        "def eval_tone_cases(): return []\ndef eval_tone(case): pass\n",
    )
    evals = load_eval_functions(f)
    assert len(evals) == 1
    assert evals[0].name == "eval_tone"
    assert evals[0].cases_func is not None
    assert callable(evals[0].cases_func)


def test_load_eval_functions_cases_func_not_returned_as_standalone(tmp_path):
    f = _write(
        tmp_path / "eval_data.py",
        "def eval_data_cases(): return []\ndef eval_data(case): pass\n",
    )
    evals = load_eval_functions(f)
    names = [e.name for e in evals]
    assert "eval_data_cases" not in names
    assert "eval_data" in names


def test_load_eval_functions_does_not_pick_up_imports(tmp_path):
    """Functions imported from other modules should not be discovered."""
    imported_module = _write(
        tmp_path / "helpers.py",
        "def eval_imported(case): pass\n",
    )
    f = _write(
        tmp_path / "eval_check.py",
        "from helpers import eval_imported\ndef eval_local(case): pass\n",
    )
    # Add tmp_path to sys.path so the import works
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        evals = load_eval_functions(f)
    finally:
        sys.path.pop(0)

    names = [e.name for e in evals]
    assert "eval_local" in names
    assert "eval_imported" not in names


# ---------------------------------------------------------------------------
# discover — integration of file search + loading
# ---------------------------------------------------------------------------


def test_discover_finds_all_funcs_in_directory(tmp_path):
    _write(tmp_path / "eval_tone.py", "def eval_tone(case): pass\n")
    _write(tmp_path / "eval_acc.py", "def eval_acc(case): pass\n")
    _write(tmp_path / "sub" / "eval_edge.py", "def eval_edge(case): pass\n")

    evals = discover(tmp_path)
    names = {e.name for e in evals}
    assert names == {"eval_tone", "eval_acc", "eval_edge"}


def test_discover_single_file(tmp_path):
    f = _write(tmp_path / "eval_tone.py", "def eval_tone(case): pass\n")
    evals = discover(f)
    assert len(evals) == 1
    assert evals[0].name == "eval_tone"


def test_discover_eval_filter_exact_match(tmp_path):
    _write(tmp_path / "eval_tone.py", "def eval_tone(case): pass\n")
    _write(tmp_path / "eval_acc.py", "def eval_acc(case): pass\n")

    evals = discover(tmp_path, eval_filter="eval_tone")
    assert len(evals) == 1
    assert evals[0].name == "eval_tone"


def test_discover_eval_filter_no_match_returns_empty(tmp_path):
    _write(tmp_path / "eval_tone.py", "def eval_tone(case): pass\n")
    evals = discover(tmp_path, eval_filter="eval_nonexistent")
    assert evals == []


def test_discover_eval_filter_finds_in_subdirectory(tmp_path):
    _write(tmp_path / "eval_top.py", "def eval_top(case): pass\n")
    _write(
        tmp_path / "sub" / "eval_edge_cases_check.py",
        "def eval_edge_cases_check(case): pass\n",
    )
    evals = discover(tmp_path, eval_filter="eval_edge_cases_check")
    assert len(evals) == 1
    assert evals[0].name == "eval_edge_cases_check"


def test_discover_empty_directory_returns_empty(tmp_path):
    assert discover(tmp_path) == []


# ---------------------------------------------------------------------------
# Naming edge-cases — functions whose names look like cases generators
# ---------------------------------------------------------------------------


def test_eval_name_ending_in_cases_without_sibling_is_standalone(tmp_path):
    """eval_edge_cases with no sibling eval_edge is a standalone eval, not a cases func."""
    f = _write(
        tmp_path / "eval_edge_cases.py",
        "def eval_edge_cases(case): pass\n",
    )
    evals = load_eval_functions(f)
    assert len(evals) == 1
    assert evals[0].name == "eval_edge_cases"
    assert evals[0].cases_func is None


def test_eval_name_ending_in_cases_with_sibling_is_cases_func(tmp_path):
    """eval_tone_cases IS a cases generator when eval_tone also exists."""
    f = _write(
        tmp_path / "eval_tone.py",
        "def eval_tone_cases(): return []\ndef eval_tone(case): pass\n",
    )
    evals = load_eval_functions(f)
    assert len(evals) == 1
    assert evals[0].name == "eval_tone"
    assert evals[0].cases_func is not None


def test_discover_finds_eval_edge_cases_via_filter(tmp_path):
    """--eval eval_edge_cases finds the function even though its name ends with _cases."""
    _write(tmp_path / "eval_edge_cases.py", "def eval_edge_cases(case): pass\n")
    evals = discover(tmp_path, eval_filter="eval_edge_cases")
    assert len(evals) == 1
    assert evals[0].name == "eval_edge_cases"
