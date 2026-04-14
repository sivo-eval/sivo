"""E2e tests for multi-provider support (Phase C).

Tests:
1. Same eval runs identically with two different mocked judge providers
   (AnthropicProvider + OpenAIProvider) — verifies pipeline behaviour is
   provider-agnostic.
2. Custom provider loaded via import path works end-to-end through run_session.
3. CLI --judge-provider flag rejects unknown providers with exit code 2.
4. sivo.toml [sivo] provider / [sivo.judge] provider config is picked
   up by the CLI.

No live LLM calls are made — all provider judge methods are mocked.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sivo.judge import LLMJudge, set_session_judge
from sivo.models import JudgeVerdict
from sivo.runner import run_session
from sivo.store import JsonlStore


# ---------------------------------------------------------------------------
# Shared helpers
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


def _write_jsonl(tmp_path: Path, run_id: str, outputs: list[str]) -> None:
    records_dir = tmp_path / ".sivo" / "records"
    records_dir.mkdir(parents=True)
    jsonl_path = records_dir / f"{run_id}.jsonl"
    with jsonl_path.open("w") as fh:
        for i, output in enumerate(outputs):
            fh.write(json.dumps(_make_record(run_id, f"rec-{i}", output)) + "\n")


def _run_cli(tmp_path: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "sivo.cli", *extra_args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Test 1: Same eval, two mocked providers → identical pipeline behaviour
# ---------------------------------------------------------------------------


_JUDGE_EVAL_CONTENT = """\
from sivo.judge import assert_judge

def eval_tone(case):
    assert_judge(case.output, rubric="tone")
"""


def _make_passing_verdict() -> JudgeVerdict:
    return JudgeVerdict(
        passed=True,
        reason="Passes the tone rubric.",
        evidence="polite and clear",
        suggestion=None,
    )


def _make_failing_verdict() -> JudgeVerdict:
    return JudgeVerdict(
        passed=False,
        reason="Fails the tone rubric.",
        evidence="rude span",
        suggestion="Be more polite.",
    )


@pytest.fixture(autouse=True)
def _clean_session_judge():
    """Ensure no session judge override leaks between tests."""
    prev = set_session_judge(None)
    yield
    set_session_judge(prev)


def test_anthropic_judge_provider_passes(tmp_path):
    """Eval with a mocked AnthropicProvider judge passes correctly."""
    run_id = "run_anthropic_test"
    (tmp_path / "eval_tone.py").write_text(_JUDGE_EVAL_CONTENT)
    _write_jsonl(tmp_path, run_id, ["This is a helpful and polite response."])

    from sivo.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key="test")
    with patch.object(provider, "judge", return_value=_make_passing_verdict()):
        judge = LLMJudge(model="claude-haiku-4-5", provider=provider)
        session = run_session(
            tmp_path / "eval_tone.py",
            run_id=run_id,
            store=JsonlStore(tmp_path / ".sivo"),
            fail_fast=False,
            judge=judge,
        )

    assert session.passed_count == 1
    assert session.failed_count == 0


def test_openai_judge_provider_passes(tmp_path):
    """Eval with a mocked OpenAIProvider judge passes correctly."""
    run_id = "run_openai_test"
    (tmp_path / "eval_tone.py").write_text(_JUDGE_EVAL_CONTENT)
    _write_jsonl(tmp_path, run_id, ["This is a helpful and polite response."])

    from sivo.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="test")
    with patch.object(provider, "judge", return_value=_make_passing_verdict()):
        judge = LLMJudge(model="gpt-4o-mini", provider=provider)
        session = run_session(
            tmp_path / "eval_tone.py",
            run_id=run_id,
            store=JsonlStore(tmp_path / ".sivo"),
            fail_fast=False,
            judge=judge,
        )

    assert session.passed_count == 1
    assert session.failed_count == 0


def test_both_providers_produce_same_session_structure(tmp_path):
    """Both providers produce structurally identical session results."""
    run_id = "run_parity"
    (tmp_path / "eval_tone.py").write_text(_JUDGE_EVAL_CONTENT)
    _write_jsonl(tmp_path, run_id, [
        "Good response.",
        "Bad response.",
        "Another good response.",
    ])

    store = JsonlStore(tmp_path / ".sivo")
    verdicts = [
        _make_passing_verdict(),
        _make_failing_verdict(),
        _make_passing_verdict(),
    ]

    from sivo.providers.anthropic import AnthropicProvider
    from sivo.providers.openai import OpenAIProvider

    results_by_provider = {}
    for pname, provider_cls in [
        ("anthropic", lambda: AnthropicProvider(api_key="test")),
        ("openai", lambda: OpenAIProvider(api_key="test")),
    ]:
        provider = provider_cls()
        call_count = 0

        def _judge_side_effect(**kwargs):
            nonlocal call_count
            v = verdicts[call_count]
            call_count += 1
            return v

        with patch.object(provider, "judge", side_effect=_judge_side_effect):
            judge = LLMJudge(model="claude-haiku-4-5", provider=provider)
            session = run_session(
                tmp_path / "eval_tone.py",
                run_id=run_id,
                store=store,
                fail_fast=False,
                judge=judge,
            )

        results_by_provider[pname] = {
            "passed": session.passed_count,
            "failed": session.failed_count,
            "total": len(session.results),
        }

    # Both providers should produce the same structure
    assert results_by_provider["anthropic"] == results_by_provider["openai"]
    assert results_by_provider["anthropic"] == {"passed": 2, "failed": 1, "total": 3}


def test_failing_judge_verdict_raises_eval_assertion_error(tmp_path):
    """Failing judge verdict from a non-default provider correctly fails the eval."""
    run_id = "run_fail_test"
    (tmp_path / "eval_tone.py").write_text(_JUDGE_EVAL_CONTENT)
    _write_jsonl(tmp_path, run_id, ["This is a rude response."])

    from sivo.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="test")
    with patch.object(provider, "judge", return_value=_make_failing_verdict()):
        judge = LLMJudge(model="gpt-4o-mini", provider=provider)
        session = run_session(
            tmp_path / "eval_tone.py",
            run_id=run_id,
            store=JsonlStore(tmp_path / ".sivo"),
            fail_fast=False,
            judge=judge,
        )

    assert session.failed_count == 1
    assert session.passed_count == 0


def test_session_judge_restored_after_run(tmp_path):
    """Session judge override is always restored after run_session completes."""
    run_id = "run_restore"
    (tmp_path / "eval_tone.py").write_text(_JUDGE_EVAL_CONTENT)
    _write_jsonl(tmp_path, run_id, ["response"])

    from sivo.judge import _get_default_judge
    from sivo.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key="test")
    custom_judge = LLMJudge(model="claude-haiku-4-5", provider=provider)

    with patch.object(provider, "judge", return_value=_make_passing_verdict()):
        run_session(
            tmp_path / "eval_tone.py",
            run_id=run_id,
            store=JsonlStore(tmp_path / ".sivo"),
            fail_fast=False,
            judge=custom_judge,
        )

    # After run_session, the session override should be cleared
    from sivo.judge import _session_judge_override
    assert _session_judge_override is None


# ---------------------------------------------------------------------------
# Test 2: Custom provider via import path through run_session
# ---------------------------------------------------------------------------

_CUSTOM_PROVIDER_MODULE = """\
from sivo.models import JudgeVerdict
from sivo.providers import CompletionResult

class AlwaysPassJudge:
    name = 'always_pass'

    def __init__(self, api_key=None):
        self.call_count = 0

    async def complete(self, *, model, system_prompt, messages,
                       max_tokens=1024, timeout=30.0, extra_params=None):
        return CompletionResult(
            output='custom response', input_tokens=1,
            output_tokens=1, cost_usd=0.0, model=model,
        )

    def judge(self, *, model, system_prompt, messages, rubric_name):
        self.call_count += 1
        return JudgeVerdict(
            passed=True,
            reason='Custom provider always passes.',
            evidence='<always pass>',
            suggestion=None,
        )
"""


def test_custom_provider_via_import_path(tmp_path, monkeypatch):
    """Custom provider loaded via 'module:ClassName' works through run_session."""
    # Write the custom provider module
    (tmp_path / "always_pass_provider.py").write_text(_CUSTOM_PROVIDER_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("always_pass_provider", None)

    run_id = "run_custom_provider"
    (tmp_path / "eval_tone.py").write_text(_JUDGE_EVAL_CONTENT)
    _write_jsonl(tmp_path, run_id, ["test response A", "test response B"])

    from sivo.providers.registry import get_provider

    provider = get_provider("always_pass_provider:AlwaysPassJudge")
    assert provider.name == "always_pass"

    judge = LLMJudge(model="custom-model", provider=provider)
    session = run_session(
        tmp_path / "eval_tone.py",
        run_id=run_id,
        store=JsonlStore(tmp_path / ".sivo"),
        fail_fast=False,
        judge=judge,
    )

    assert session.passed_count == 2
    assert session.failed_count == 0
    # Verify the custom provider's judge was actually called
    assert provider.call_count == 2


# ---------------------------------------------------------------------------
# Test 3: CLI --judge-provider flag
# ---------------------------------------------------------------------------

_SIMPLE_EVAL_CONTENT = """\
from sivo.assertions import assert_contains

def eval_check(case):
    assert_contains(case.output, "good")
"""


def test_cli_judge_provider_unknown_exits_two(tmp_path):
    """Unknown --judge-provider exits with code 2 and an error message."""
    run_id = "run_bad_provider"
    (tmp_path / "eval_check.py").write_text(_SIMPLE_EVAL_CONTENT)
    _write_jsonl(tmp_path, run_id, ["good response"])

    result = _run_cli(
        tmp_path,
        "run", "eval_check.py",
        "--run-id", run_id,
        "--judge-provider", "no-such-provider",
    )

    assert result.returncode == 2
    assert "no-such-provider" in result.stderr


def test_cli_judge_provider_anthropic_explicit_exits_zero(tmp_path):
    """Explicit --judge-provider anthropic is accepted (deterministic eval, no judge call)."""
    run_id = "run_explicit_anthropic"
    (tmp_path / "eval_check.py").write_text(_SIMPLE_EVAL_CONTENT)
    _write_jsonl(tmp_path, run_id, ["good response"])

    result = _run_cli(
        tmp_path,
        "run", "eval_check.py",
        "--run-id", run_id,
        "--judge-provider", "anthropic",
    )

    assert result.returncode == 0, (
        f"Expected exit 0 but got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_cli_judge_model_flag_accepted(tmp_path):
    """--judge-model flag is accepted (deterministic eval, no actual judge call)."""
    run_id = "run_judge_model"
    (tmp_path / "eval_check.py").write_text(_SIMPLE_EVAL_CONTENT)
    _write_jsonl(tmp_path, run_id, ["good response"])

    result = _run_cli(
        tmp_path,
        "run", "eval_check.py",
        "--run-id", run_id,
        "--judge-model", "claude-sonnet-4-6",
    )

    assert result.returncode == 0, (
        f"Expected exit 0 but got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 4: sivo.toml provider config is picked up by the CLI
# ---------------------------------------------------------------------------


def test_toml_provider_config_accepted(tmp_path):
    """sivo.toml with provider = 'anthropic' is accepted by the CLI."""
    run_id = "run_toml_provider"
    (tmp_path / "eval_check.py").write_text(_SIMPLE_EVAL_CONTENT)
    _write_jsonl(tmp_path, run_id, ["good response"])
    (tmp_path / "sivo.toml").write_text(
        "[sivo]\n"
        "provider = 'anthropic'\n"
        "\n"
        "[sivo.judge]\n"
        "default_model = 'claude-haiku-4-5'\n"
    )

    result = _run_cli(
        tmp_path,
        "run", "eval_check.py",
        "--run-id", run_id,
    )

    assert result.returncode == 0, (
        f"Expected exit 0 but got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_toml_judge_provider_unknown_exits_two(tmp_path):
    """sivo.toml with an unknown judge provider causes exit 2."""
    run_id = "run_toml_bad_judge"
    (tmp_path / "eval_check.py").write_text(_SIMPLE_EVAL_CONTENT)
    _write_jsonl(tmp_path, run_id, ["good response"])
    (tmp_path / "sivo.toml").write_text(
        "[sivo.judge]\n"
        "provider = 'no-such-provider'\n"
    )

    result = _run_cli(
        tmp_path,
        "run", "eval_check.py",
        "--run-id", run_id,
    )

    assert result.returncode == 2
    assert "no-such-provider" in result.stderr
