"""E2e packaging smoke test.

Builds the sivo wheel (or uses a pre-built one), installs it into a fresh
virtual environment, then runs ``sivo run`` inside that venv against a
minimal data-driven eval file to confirm the installed package works end-to-end.

This test is marked ``packaging`` and is excluded from the default ``pytest``
run to avoid the build cost on every CI push.  Run explicitly with:

    uv run pytest tests/e2e/test_packaging.py -m packaging -v

The test requires ``uv build`` to be available (it is present in the developer
environment via the ``uv`` tool used by this project).
"""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_or_build_wheel(repo_root: Path) -> Path:
    """Return the newest wheel in dist/, building it first if needed."""
    dist = repo_root / "dist"
    wheels = sorted(dist.glob("sivo-*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if wheels:
        return wheels[0]

    # Build the wheel
    result = subprocess.run(
        [sys.executable, "-m", "uv", "build", "--wheel"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Try uv build directly
        result2 = subprocess.run(
            ["uv", "build", "--wheel"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result2.returncode != 0:
            pytest.skip(f"Could not build wheel: {result2.stderr}")

    wheels = sorted(dist.glob("sivo-*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wheels:
        pytest.skip("Wheel build produced no .whl file")
    return wheels[0]


def _make_venv(venv_path: Path) -> Path:
    """Create a fresh venv and return its Python executable path."""
    venv.create(str(venv_path), with_pip=True, clear=True)
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _pip_install(python: Path, *packages: str) -> None:
    """Install packages into the venv silently."""
    result = subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", *packages],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"pip install failed:\n{result.stderr}"


# ---------------------------------------------------------------------------
# The smoke test
# ---------------------------------------------------------------------------


@pytest.mark.packaging
def test_installed_package_runs_data_driven_eval(tmp_path):
    """Install sivo wheel into a fresh venv and run a minimal eval."""
    repo_root = Path(__file__).parent.parent.parent

    # 1. Find or build the wheel
    wheel = _find_or_build_wheel(repo_root)

    # 2. Create a fresh virtual environment
    venv_path = tmp_path / "test-venv"
    python = _make_venv(venv_path)

    # 3. Install sivo (the wheel) + its runtime dependencies
    _pip_install(python, str(wheel))

    # 4. Write a minimal data-driven eval file in tmp_path
    eval_file = tmp_path / "eval_smoke.py"
    eval_file.write_text(
        "from sivo.models import EvalCase\n"
        "from sivo.assertions import assert_contains\n"
        "\n"
        "def eval_smoke_cases():\n"
        "    return [\n"
        "        EvalCase(input='hi', output='hello world'),\n"
        "        EvalCase(input='greet', output='welcome to sivo'),\n"
        "    ]\n"
        "\n"
        "def eval_smoke(case):\n"
        "    # At least one of 'hello', 'welcome', 'sivo' must be in output\n"
        "    found = any(w in case.output for w in ['hello', 'welcome', 'sivo'])\n"
        "    assert found, f'Unexpected output: {case.output!r}'\n"
    )

    # 5. Run ``sivo run`` via the installed CLI
    sivo_cli = (
        venv_path / "bin" / "sivo"
        if sys.platform != "win32"
        else venv_path / "Scripts" / "sivo.exe"
    )
    result = subprocess.run(
        [str(sivo_cli), "run", str(eval_file),
         "--store-path", str(tmp_path / ".sivo")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    # 6. Assert success
    output = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"sivo exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # Session receipt should appear in output
    assert "sivo run complete" in output, f"Receipt not found in output:\n{output}"
    assert "2 passed" in output, f"Expected '2 passed' in output:\n{output}"


@pytest.mark.packaging
def test_installed_package_imports_cleanly(tmp_path):
    """All public API symbols are importable from the installed package."""
    repo_root = Path(__file__).parent.parent.parent
    wheel = _find_or_build_wheel(repo_root)

    venv_path = tmp_path / "test-venv"
    python = _make_venv(venv_path)
    _pip_install(python, str(wheel))

    import_check = (
        "import sivo\n"
        "from sivo import (\n"
        "    fixture, get_response,\n"
        "    EvalCase, ExecutionRecord, JudgeVerdict,\n"
        "    EvalResult, SessionResult,\n"
        "    assert_contains, assert_not_contains, assert_regex,\n"
        "    assert_length, assert_matches_schema,\n"
        "    EvalAssertionError, FlakyEvalError,\n"
        "    SivoConfig, load_config,\n"
        ")\n"
        "print('OK')\n"
    )

    result = subprocess.run(
        [str(python), "-c", import_check],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import failed:\n{result.stderr}"
    assert result.stdout.strip() == "OK"
