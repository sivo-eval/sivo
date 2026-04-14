"""Unit tests for sivo.providers — protocol, CompletionResult, registry, and AnthropicProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sivo.models import JudgeVerdict
from sivo.providers import CompletionResult, Provider
from sivo.providers.anthropic import (
    AnthropicProvider,
    _calculate_cost,
    _build_judge_tool,
)
from sivo.providers.registry import get_provider


# ---------------------------------------------------------------------------
# CompletionResult
# ---------------------------------------------------------------------------


def test_completion_result_is_frozen():
    r = CompletionResult(output="hi", input_tokens=1, output_tokens=2, cost_usd=0.0, model="m")
    with pytest.raises((AttributeError, TypeError)):
        r.output = "changed"  # type: ignore[misc]


def test_completion_result_equality():
    r1 = CompletionResult(output="hi", input_tokens=1, output_tokens=2, cost_usd=0.0, model="m")
    r2 = CompletionResult(output="hi", input_tokens=1, output_tokens=2, cost_usd=0.0, model="m")
    assert r1 == r2


def test_completion_result_raw_response_excluded_from_equality():
    r1 = CompletionResult(output="hi", input_tokens=1, output_tokens=2, cost_usd=0.0, model="m", raw_response={"a": 1})
    r2 = CompletionResult(output="hi", input_tokens=1, output_tokens=2, cost_usd=0.0, model="m", raw_response={"b": 2})
    assert r1 == r2  # raw_response has compare=False


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


def test_anthropic_provider_satisfies_protocol():
    provider = AnthropicProvider(api_key="test")
    assert isinstance(provider, Provider)


def test_provider_protocol_has_name():
    provider = AnthropicProvider(api_key="test")
    assert provider.name == "anthropic"


def test_provider_protocol_has_complete_and_judge():
    provider = AnthropicProvider(api_key="test")
    assert callable(getattr(provider, "complete", None))
    assert callable(getattr(provider, "judge", None))


# ---------------------------------------------------------------------------
# AnthropicProvider — _calculate_cost
# ---------------------------------------------------------------------------


def test_calculate_cost_known_model():
    cost = _calculate_cost("claude-haiku-4-5", 1_000_000, 1_000_000)
    assert cost == pytest.approx(4.80)  # 0.80 + 4.00


def test_calculate_cost_zero_tokens():
    assert _calculate_cost("claude-haiku-4-5", 0, 0) == 0.0


def test_calculate_cost_unknown_model_uses_fallback():
    # Falls back to Sonnet pricing: (3.00 + 15.00) / 1M
    cost = _calculate_cost("unknown-model", 1_000_000, 1_000_000)
    assert cost == pytest.approx(18.00)


# ---------------------------------------------------------------------------
# AnthropicProvider — _build_judge_tool
# ---------------------------------------------------------------------------


def test_build_judge_tool_structure():
    tool = _build_judge_tool()
    assert tool["name"] == "record_verdict"
    assert "input_schema" in tool
    props = tool["input_schema"]["properties"]
    assert "passed" in props
    assert "reason" in props
    assert "evidence" in props
    assert "suggestion" in props
    assert "passed" in tool["input_schema"]["required"]


# ---------------------------------------------------------------------------
# AnthropicProvider — complete()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_provider_complete_basic():
    provider = AnthropicProvider(api_key="test-key")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Answer text")]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.complete(
            model="claude-haiku-4-5",
            system_prompt=None,
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert isinstance(result, CompletionResult)
    assert result.output == "Answer text"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.model == "claude-haiku-4-5"
    assert result.cost_usd == pytest.approx(_calculate_cost("claude-haiku-4-5", 10, 5))


@pytest.mark.asyncio
async def test_anthropic_provider_complete_with_system_prompt():
    provider = AnthropicProvider(api_key="test-key")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="response")]
    mock_response.usage = MagicMock(input_tokens=5, output_tokens=3)

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        await provider.complete(
            model="claude-haiku-4-5",
            system_prompt="You are helpful.",
            messages=[{"role": "user", "content": "Hi"}],
        )

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "You are helpful."


@pytest.mark.asyncio
async def test_anthropic_provider_complete_no_system_prompt():
    provider = AnthropicProvider(api_key="test-key")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="response")]
    mock_response.usage = MagicMock(input_tokens=5, output_tokens=3)

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        await provider.complete(
            model="claude-haiku-4-5",
            system_prompt=None,
            messages=[{"role": "user", "content": "Hi"}],
        )

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "system" not in call_kwargs


@pytest.mark.asyncio
async def test_anthropic_provider_complete_extra_params():
    provider = AnthropicProvider(api_key="test-key")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="response")]
    mock_response.usage = MagicMock(input_tokens=5, output_tokens=3)

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        await provider.complete(
            model="claude-haiku-4-5",
            system_prompt=None,
            messages=[{"role": "user", "content": "Hi"}],
            extra_params={"temperature": 0.7, "max_tokens": 256},
        )

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 256


@pytest.mark.asyncio
async def test_anthropic_provider_complete_empty_content():
    """Empty response content yields empty string output."""
    provider = AnthropicProvider(api_key="test-key")

    mock_response = MagicMock()
    mock_response.content = []
    mock_response.usage = MagicMock(input_tokens=5, output_tokens=0)

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.complete(
            model="claude-haiku-4-5",
            system_prompt=None,
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert result.output == ""


# ---------------------------------------------------------------------------
# AnthropicProvider — judge()
# ---------------------------------------------------------------------------


def test_anthropic_provider_judge_basic():
    provider = AnthropicProvider(api_key="test-key")

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "record_verdict"
    tool_block.input = {
        "passed": True,
        "reason": "Passes the rubric.",
        "evidence": "some text",
        "suggestion": None,
    }

    mock_response = MagicMock()
    mock_response.content = [tool_block]

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = MagicMock(return_value=mock_response)

        verdict = provider.judge(
            model="claude-haiku-4-5",
            system_prompt="You are a judge.",
            messages=[{"role": "user", "content": "Evaluate this."}],
            rubric_name="helpfulness",
        )

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.passed is True
    assert verdict.reason == "Passes the rubric."


def test_anthropic_provider_judge_raises_on_missing_tool_call():
    provider = AnthropicProvider(api_key="test-key")

    non_tool_block = MagicMock()
    non_tool_block.type = "text"

    mock_response = MagicMock()
    mock_response.content = [non_tool_block]

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = MagicMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="record_verdict"):
            provider.judge(
                model="claude-haiku-4-5",
                system_prompt="You are a judge.",
                messages=[{"role": "user", "content": "Evaluate."}],
                rubric_name="helpfulness",
            )


def test_anthropic_provider_judge_passes_tool_choice():
    """Anthropic judge call must force the record_verdict tool."""
    provider = AnthropicProvider(api_key="test-key")

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "record_verdict"
    tool_block.input = {"passed": True, "reason": "ok", "evidence": "e"}

    mock_response = MagicMock()
    mock_response.content = [tool_block]

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = MagicMock(return_value=mock_response)

        provider.judge(
            model="claude-haiku-4-5",
            system_prompt="system",
            messages=[{"role": "user", "content": "msg"}],
            rubric_name="tone",
        )

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "record_verdict"}


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


def test_get_provider_anthropic():
    p = get_provider("anthropic", api_key="test")
    assert isinstance(p, AnthropicProvider)
    assert p.name == "anthropic"


def test_get_provider_unknown_builtin_raises():
    with pytest.raises(ValueError, match="no-such-provider"):
        get_provider("no-such-provider")


def test_get_provider_invalid_name_no_colon():
    with pytest.raises(ValueError, match="custom_unknown"):
        get_provider("custom_unknown")


def test_get_provider_custom_import_path(tmp_path, monkeypatch):
    """Custom providers load via 'module:ClassName' import path."""
    import sys

    # Create a minimal provider module
    mod_file = tmp_path / "my_provider.py"
    mod_file.write_text(
        "from sivo.models import JudgeVerdict\n"
        "from sivo.providers import CompletionResult\n"
        "\n"
        "class MyProvider:\n"
        "    name = 'my_provider'\n"
        "\n"
        "    def __init__(self, api_key=None): pass\n"
        "\n"
        "    async def complete(self, *, model, system_prompt, messages,\n"
        "                       max_tokens=1024, timeout=30.0, extra_params=None):\n"
        "        return CompletionResult(output='hi', input_tokens=1,\n"
        "                               output_tokens=1, cost_usd=0.0, model=model)\n"
        "\n"
        "    def judge(self, *, model, system_prompt, messages, rubric_name):\n"
        "        return JudgeVerdict(passed=True, reason='ok', evidence='e')\n"
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    # Remove cached module if any
    sys.modules.pop("my_provider", None)

    p = get_provider("my_provider:MyProvider")
    assert p.name == "my_provider"
    assert isinstance(p, Provider)


def test_get_provider_custom_not_protocol_raises(tmp_path, monkeypatch):
    """A class missing protocol methods raises TypeError."""
    import sys

    mod_file = tmp_path / "bad_provider.py"
    mod_file.write_text(
        "class BadProvider:\n"
        "    def __init__(self, api_key=None): pass\n"
        "    # missing name, complete, judge\n"
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("bad_provider", None)

    with pytest.raises(TypeError, match="Provider protocol"):
        get_provider("bad_provider:BadProvider")
