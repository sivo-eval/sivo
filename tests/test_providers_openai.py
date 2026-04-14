"""Unit tests for sivo.providers.openai — OpenAIProvider and helpers."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sivo.models import JudgeVerdict
from sivo.providers import CompletionResult, Provider
from sivo.providers.openai import (
    OpenAIProvider,
    _build_judge_tool,
    _calculate_cost,
)
from sivo.providers.registry import get_provider


# ---------------------------------------------------------------------------
# OpenAIProvider — protocol conformance
# ---------------------------------------------------------------------------


def test_openai_provider_satisfies_protocol():
    provider = OpenAIProvider(api_key="test")
    assert isinstance(provider, Provider)


def test_provider_name():
    provider = OpenAIProvider(api_key="test")
    assert provider.name == "openai"


def test_provider_has_complete_and_judge():
    provider = OpenAIProvider(api_key="test")
    assert callable(getattr(provider, "complete", None))
    assert callable(getattr(provider, "judge", None))


# ---------------------------------------------------------------------------
# _calculate_cost
# ---------------------------------------------------------------------------


def test_calculate_cost_known_model():
    # gpt-4o: 2.50 + 10.00 per 1M tokens
    cost = _calculate_cost("gpt-4o", 1_000_000, 1_000_000)
    assert cost == pytest.approx(12.50)


def test_calculate_cost_mini_model():
    # gpt-4o-mini: 0.15 + 0.60 per 1M tokens
    cost = _calculate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.75)


def test_calculate_cost_zero_tokens():
    assert _calculate_cost("gpt-4o", 0, 0) == 0.0


def test_calculate_cost_unknown_model_uses_fallback():
    # Falls back to gpt-4o pricing: (2.50 + 10.00) / 1M
    cost = _calculate_cost("unknown-model", 1_000_000, 1_000_000)
    assert cost == pytest.approx(12.50)


# ---------------------------------------------------------------------------
# _build_judge_tool
# ---------------------------------------------------------------------------


def test_build_judge_tool_structure():
    tool = _build_judge_tool()
    assert tool["type"] == "function"
    func = tool["function"]
    assert func["name"] == "record_verdict"
    assert "parameters" in func
    props = func["parameters"]["properties"]
    assert "passed" in props
    assert "reason" in props
    assert "evidence" in props
    assert "suggestion" in props
    assert "passed" in func["parameters"]["required"]


# ---------------------------------------------------------------------------
# OpenAIProvider — complete()
# ---------------------------------------------------------------------------


def _make_openai_completion_response(
    content: str,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    model: str = "gpt-4o",
) -> MagicMock:
    """Build a mock openai ChatCompletion response."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


@pytest.mark.asyncio
async def test_openai_provider_complete_basic():
    provider = OpenAIProvider(api_key="test-key")
    mock_response = _make_openai_completion_response("Answer text", 10, 5)

    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.complete(
            model="gpt-4o",
            system_prompt=None,
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert isinstance(result, CompletionResult)
    assert result.output == "Answer text"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.model == "gpt-4o"
    assert result.cost_usd == pytest.approx(_calculate_cost("gpt-4o", 10, 5))


@pytest.mark.asyncio
async def test_openai_provider_complete_with_system_prompt():
    provider = OpenAIProvider(api_key="test-key")
    mock_response = _make_openai_completion_response("response")

    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        await provider.complete(
            model="gpt-4o",
            system_prompt="You are helpful.",
            messages=[{"role": "user", "content": "Hi"}],
        )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "You are helpful."}
    assert messages[1] == {"role": "user", "content": "Hi"}


@pytest.mark.asyncio
async def test_openai_provider_complete_no_system_prompt():
    provider = OpenAIProvider(api_key="test-key")
    mock_response = _make_openai_completion_response("response")

    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        await provider.complete(
            model="gpt-4o",
            system_prompt=None,
            messages=[{"role": "user", "content": "Hi"}],
        )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    # No system message prepended when system_prompt is None
    assert messages[0]["role"] == "user"
    assert len(messages) == 1


@pytest.mark.asyncio
async def test_openai_provider_complete_extra_params():
    provider = OpenAIProvider(api_key="test-key")
    mock_response = _make_openai_completion_response("response")

    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        await provider.complete(
            model="gpt-4o",
            system_prompt=None,
            messages=[{"role": "user", "content": "Hi"}],
            extra_params={"temperature": 0.7},
        )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.7


@pytest.mark.asyncio
async def test_openai_provider_complete_null_content_yields_empty_string():
    """When message.content is None, output should be ''."""
    provider = OpenAIProvider(api_key="test-key")
    mock_response = _make_openai_completion_response(None)  # type: ignore[arg-type]

    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.complete(
            model="gpt-4o",
            system_prompt=None,
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert result.output == ""


# ---------------------------------------------------------------------------
# OpenAIProvider — judge()
# ---------------------------------------------------------------------------


def _make_tool_call(args: dict) -> MagicMock:
    """Build a mock tool_call object."""
    function = MagicMock()
    function.name = "record_verdict"
    function.arguments = json.dumps(args)

    tool_call = MagicMock()
    tool_call.function = function
    return tool_call


def _make_judge_response(args: dict) -> MagicMock:
    """Build a mock openai response with a function-calling tool_call."""
    message = MagicMock()
    message.tool_calls = [_make_tool_call(args)]

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


def test_openai_provider_judge_basic():
    provider = OpenAIProvider(api_key="test-key")
    mock_response = _make_judge_response({
        "passed": True,
        "reason": "Passes the rubric.",
        "evidence": "some text",
        "suggestion": None,
    })

    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)

        verdict = provider.judge(
            model="gpt-4o",
            system_prompt="You are a judge.",
            messages=[{"role": "user", "content": "Evaluate this."}],
            rubric_name="helpfulness",
        )

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.passed is True
    assert verdict.reason == "Passes the rubric."
    assert verdict.evidence == "some text"
    assert verdict.suggestion is None


def test_openai_provider_judge_failed_verdict():
    provider = OpenAIProvider(api_key="test-key")
    mock_response = _make_judge_response({
        "passed": False,
        "reason": "Fails the rubric.",
        "evidence": "bad span",
        "suggestion": "Try harder.",
    })

    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)

        verdict = provider.judge(
            model="gpt-4o",
            system_prompt="You are a judge.",
            messages=[{"role": "user", "content": "Evaluate."}],
            rubric_name="tone",
        )

    assert verdict.passed is False
    assert verdict.suggestion == "Try harder."


def test_openai_provider_judge_raises_on_missing_tool_call():
    provider = OpenAIProvider(api_key="test-key")

    message = MagicMock()
    message.tool_calls = None  # No tool calls returned

    choice = MagicMock()
    choice.message = message

    mock_response = MagicMock()
    mock_response.choices = [choice]

    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="record_verdict"):
            provider.judge(
                model="gpt-4o",
                system_prompt="You are a judge.",
                messages=[{"role": "user", "content": "Evaluate."}],
                rubric_name="helpfulness",
            )


def test_openai_provider_judge_passes_tool_choice():
    """OpenAI judge call must force the record_verdict function."""
    provider = OpenAIProvider(api_key="test-key")
    mock_response = _make_judge_response({
        "passed": True,
        "reason": "ok",
        "evidence": "e",
    })

    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)

        provider.judge(
            model="gpt-4o",
            system_prompt="system",
            messages=[{"role": "user", "content": "msg"}],
            rubric_name="tone",
        )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "record_verdict"},
    }


def test_openai_provider_judge_system_prompt_prepended():
    """System prompt must be the first message in the judge call."""
    provider = OpenAIProvider(api_key="test-key")
    mock_response = _make_judge_response({
        "passed": True,
        "reason": "ok",
        "evidence": "e",
    })

    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)

        provider.judge(
            model="gpt-4o",
            system_prompt="Judge carefully.",
            messages=[{"role": "user", "content": "msg"}],
            rubric_name="tone",
        )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "Judge carefully."}


# ---------------------------------------------------------------------------
# Provider registry — openai integration
# ---------------------------------------------------------------------------


def test_get_provider_openai():
    p = get_provider("openai", api_key="test")
    assert isinstance(p, OpenAIProvider)
    assert p.name == "openai"


def test_get_provider_openai_no_key():
    """get_provider('openai') succeeds even without an api_key (key may come from env)."""
    p = get_provider("openai")
    assert isinstance(p, OpenAIProvider)


# ---------------------------------------------------------------------------
# Integration tests (skipped with --no-llm or missing OPENAI_API_KEY)
# ---------------------------------------------------------------------------


@pytest.fixture
def no_llm(request):
    return request.config.getoption("--no-llm", default=False)


@pytest.mark.integration
class TestOpenAIProviderIntegration:
    """Live tests that hit the OpenAI API.

    Skip with: uv run pytest --no-llm
    Requires: OPENAI_API_KEY environment variable set.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_llm_or_no_key(self, no_llm):
        import os

        if no_llm:
            pytest.skip("--no-llm flag set; skipping live LLM tests.")
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set; skipping live OpenAI tests.")

    @pytest.mark.asyncio
    async def test_complete_returns_text(self):
        provider = OpenAIProvider()
        result = await provider.complete(
            model="gpt-4o-mini",
            system_prompt=None,
            messages=[{"role": "user", "content": "Reply with the single word 'hello'."}],
            max_tokens=16,
        )
        assert isinstance(result, CompletionResult)
        assert result.output.strip()
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.cost_usd >= 0.0

    def test_judge_returns_verdict(self):
        from sivo.judge import BUILTIN_RUBRICS

        provider = OpenAIProvider()
        rubric_text = BUILTIN_RUBRICS["helpfulness"]
        verdict = provider.judge(
            model="gpt-4o-mini",
            system_prompt=(
                "You are an evaluation judge. "
                f"Assess whether the response meets this rubric:\n{rubric_text}"
            ),
            messages=[{
                "role": "user",
                "content": (
                    "To cancel your subscription, go to Settings → Billing → Cancel Plan."
                ),
            }],
            rubric_name="helpfulness",
        )
        assert isinstance(verdict, JudgeVerdict)
        assert verdict.reason
        assert verdict.evidence
