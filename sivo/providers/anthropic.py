"""Anthropic provider implementation for sivo.

Wraps the Anthropic SDK's async completion and synchronous judge calls.
This module contains all Anthropic-specific code so that the execution engine
and judge remain provider-agnostic.
"""

from __future__ import annotations

import os
from typing import Any

from sivo.models import JudgeVerdict
from sivo.providers import CompletionResult

# ---------------------------------------------------------------------------
# Cost table (USD per 1M tokens) — Anthropic models (D-012)
# ---------------------------------------------------------------------------

_ANTHROPIC_COST_TABLE: dict[str, tuple[float, float]] = {
    # model: (input_per_1m_usd, output_per_1m_usd)
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-6-20251101": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
    "claude-opus-4-6-20251101": (15.00, 75.00),
    # Older / generic aliases kept for compatibility
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
}

_FALLBACK_COST = (3.00, 15.00)  # Sonnet pricing as a safe fallback


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for Anthropic *model* given token counts.

    Public so that it can be re-exported from ``sivo.runner`` for
    backwards-compatibility.
    """
    in_rate, out_rate = _ANTHROPIC_COST_TABLE.get(model, _FALLBACK_COST)
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


# ---------------------------------------------------------------------------
# Judge tool definition
# ---------------------------------------------------------------------------


def _build_judge_tool() -> dict[str, Any]:
    """Build the Anthropic tool definition for structured judge output."""
    return {
        "name": "record_verdict",
        "description": "Record the evaluation verdict as structured output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "passed": {
                    "type": "boolean",
                    "description": "True if the response passes the rubric, False otherwise.",
                },
                "reason": {
                    "type": "string",
                    "description": "One sentence explaining the verdict.",
                },
                "evidence": {
                    "type": "string",
                    "description": (
                        "A direct quote from the model response that most strongly "
                        "influenced the verdict. If no relevant span exists, "
                        "write '<no direct evidence>'."
                    ),
                },
                "suggestion": {
                    "type": "string",
                    "description": (
                        "Optional one-sentence suggestion for how to improve the "
                        "response. Omit if the response passed."
                    ),
                },
            },
            "required": ["passed", "reason", "evidence"],
        },
    }


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


class AnthropicProvider:
    """LLM provider backed by the Anthropic API.

    All Anthropic SDK imports are deferred to method bodies so that the module
    can be imported without the SDK installed (for testing or offline use).

    Args:
        api_key: Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY``
                 environment variable if not supplied.
    """

    name: str = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    # ------------------------------------------------------------------
    # Provider protocol implementation
    # ------------------------------------------------------------------

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: list[dict],
        max_tokens: int = 1024,
        timeout: float = 30.0,
        extra_params: dict | None = None,
    ) -> CompletionResult:
        """Call ``client.messages.create`` and return a :class:`~sivo.providers.CompletionResult`.

        Args:
            model:         Exact Anthropic model string.
            system_prompt: Optional system prompt passed as the ``system`` field.
            messages:      Pre-formatted message dicts (``role`` + ``content``).
            max_tokens:    Token budget for the response.
            timeout:       Not used directly here — caller wraps with
                           ``asyncio.wait_for``.
            extra_params:  Arbitrary kwargs merged into the API call (e.g.
                           ``{"temperature": 0.7}``).

        Returns:
            :class:`~sivo.providers.CompletionResult` with normalised fields.
        """
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)

        call_kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system_prompt:
            call_kwargs["system"] = system_prompt

        if extra_params:
            call_kwargs.update(extra_params)

        response = await client.messages.create(**call_kwargs)

        output = response.content[0].text if response.content else ""
        input_tokens: int = response.usage.input_tokens
        output_tokens: int = response.usage.output_tokens
        cost_usd = _calculate_cost(model, input_tokens, output_tokens)

        return CompletionResult(
            output=output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            model=model,
        )

    def judge(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[dict],
        rubric_name: str,
    ) -> JudgeVerdict:
        """Call the Anthropic API with ``tool_use`` to extract a structured verdict.

        Args:
            model:         Exact Anthropic model string for the judge.
            system_prompt: Pre-formatted system prompt from :class:`~sivo.judge.LLMJudge`.
            messages:      Pre-formatted user message(s) from :class:`~sivo.judge.LLMJudge`.
            rubric_name:   Rubric identifier (used only in error messages).

        Returns:
            :class:`~sivo.models.JudgeVerdict` parsed from the ``record_verdict``
            tool call in the response.

        Raises:
            RuntimeError: If the model does not return a ``record_verdict`` tool call.
        """
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        tool_def = _build_judge_tool()

        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=system_prompt,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": "record_verdict"},
            messages=messages,
        )

        tool_input = self._extract_tool_input(response, rubric_name)
        return JudgeVerdict(**tool_input)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tool_input(response: Any, rubric_name: str = "") -> dict[str, Any]:
        """Extract the ``record_verdict`` tool input from an Anthropic response."""
        for block in response.content:
            if block.type == "tool_use" and block.name == "record_verdict":
                return block.input  # type: ignore[return-value]
        raise RuntimeError(
            f"Judge model did not return a record_verdict tool call "
            f"(rubric={rubric_name!r}). Response content: {response.content}"
        )
