"""OpenAI provider implementation for sivo.

Wraps the OpenAI SDK's async completion and synchronous judge calls.
This module contains all OpenAI-specific code so that the execution engine
and judge remain provider-agnostic.

Optional dependency — install with: pip install sivo[openai]
"""

from __future__ import annotations

import json
import os
from typing import Any

from sivo.models import JudgeVerdict
from sivo.providers import CompletionResult

# ---------------------------------------------------------------------------
# Cost table (USD per 1M tokens) — OpenAI models
# ---------------------------------------------------------------------------

_OPENAI_COST_TABLE: dict[str, tuple[float, float]] = {
    # model: (input_per_1m_usd, output_per_1m_usd)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-11-20": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4-turbo-2024-04-09": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-3.5-turbo-0125": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-2024-12-17": (15.00, 60.00),
    "o1-mini": (1.10, 4.40),
    "o1-mini-2024-09-12": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
    "o3-mini-2025-01-31": (1.10, 4.40),
    "o3": (10.00, 40.00),
}

_FALLBACK_COST = (2.50, 10.00)  # gpt-4o pricing as a safe fallback


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for OpenAI *model* given token counts."""
    in_rate, out_rate = _OPENAI_COST_TABLE.get(model, _FALLBACK_COST)
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


# ---------------------------------------------------------------------------
# Judge tool definition
# ---------------------------------------------------------------------------


def _build_judge_tool() -> dict[str, Any]:
    """Build the OpenAI function definition for structured judge output."""
    return {
        "type": "function",
        "function": {
            "name": "record_verdict",
            "description": "Record the evaluation verdict as structured output.",
            "parameters": {
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
        },
    }


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


class OpenAIProvider:
    """LLM provider backed by the OpenAI API.

    All OpenAI SDK imports are deferred to method bodies so that the module
    can be imported without the SDK installed (for testing or offline use).

    Args:
        api_key:  OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
                  environment variable if not supplied.
        base_url: Optional base URL override for Azure OpenAI or proxies.
    """

    name: str = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url

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
        """Call ``client.chat.completions.create`` and return a :class:`~sivo.providers.CompletionResult`.

        The OpenAI API uses the same ``role``/``content`` message format as
        sivo's internal representation. If *system_prompt* is provided it is
        prepended as a ``{"role": "system", "content": ...}`` message.

        Args:
            model:         Exact OpenAI model string (e.g. ``"gpt-4o"``).
            system_prompt: Optional system prompt; prepended to *messages*.
            messages:      Pre-formatted message dicts (``role`` + ``content``).
            max_tokens:    Token budget for the response.
            timeout:       Per-call timeout in seconds.
            extra_params:  Arbitrary kwargs merged into the API call.

        Returns:
            :class:`~sivo.providers.CompletionResult` with normalised fields.
        """
        import openai

        client = openai.AsyncOpenAI(
            api_key=self._api_key,
            **({"base_url": self._base_url} if self._base_url else {}),
        )

        full_messages: list[dict] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        call_kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": full_messages,
            "timeout": timeout,
        }

        if extra_params:
            call_kwargs.update(extra_params)

        response = await client.chat.completions.create(**call_kwargs)

        choice = response.choices[0]
        output = choice.message.content or ""
        input_tokens: int = response.usage.prompt_tokens
        output_tokens: int = response.usage.completion_tokens
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
        """Call the OpenAI API with function calling to extract a structured verdict.

        Falls back to :mod:`sivo.providers._fallback_judge` if the model does
        not return a function call (e.g. when the model refuses or malfunctions).

        Args:
            model:         Exact OpenAI model string for the judge.
            system_prompt: Pre-formatted system prompt from :class:`~sivo.judge.LLMJudge`.
            messages:      Pre-formatted user message(s).
            rubric_name:   Rubric identifier (used only in error messages).

        Returns:
            :class:`~sivo.models.JudgeVerdict` parsed from the function call.

        Raises:
            RuntimeError: If function calling fails and the fallback also fails.
        """
        import openai

        client = openai.OpenAI(
            api_key=self._api_key,
            **({"base_url": self._base_url} if self._base_url else {}),
        )

        tool_def = _build_judge_tool()

        full_messages: list[dict] = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        response = client.chat.completions.create(
            model=model,
            max_tokens=512,
            messages=full_messages,
            tools=[tool_def],
            tool_choice={"type": "function", "function": {"name": "record_verdict"}},
        )

        return self._extract_tool_input(response, rubric_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tool_input(response: Any, rubric_name: str = "") -> JudgeVerdict:
        """Extract the ``record_verdict`` function call from an OpenAI response."""
        choice = response.choices[0]
        message = choice.message

        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "record_verdict":
                    args = json.loads(tool_call.function.arguments)
                    return JudgeVerdict(
                        passed=bool(args["passed"]),
                        reason=str(args["reason"]),
                        evidence=str(args.get("evidence", "<no direct evidence>")),
                        suggestion=str(args["suggestion"]) if args.get("suggestion") else None,
                    )

        raise RuntimeError(
            f"Judge model did not return a record_verdict function call "
            f"(rubric={rubric_name!r}). Response: {response}"
        )
