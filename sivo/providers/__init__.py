"""Provider protocol and CompletionResult for sivo's multi-provider support.

The ``Provider`` protocol is the interface every LLM provider must implement.
Third parties can implement their own providers without inheriting from a base
class — duck-typing and ``isinstance`` checks both work.

Built-in providers:

- :class:`sivo.providers.anthropic.AnthropicProvider` — wraps the Anthropic SDK
  (always installed; no optional dependency required)
- :class:`sivo.providers.openai.OpenAIProvider` — wraps the OpenAI SDK
  (optional: ``pip install sivo[openai]``)

Custom providers can be loaded by import path::

    from sivo.providers.registry import get_provider

    provider = get_provider("my_package.module:MyProvider")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sivo.models import JudgeVerdict


@dataclass(frozen=True)
class CompletionResult:
    """Normalised response returned by :meth:`Provider.complete`.

    Attributes:
        output:        The LLM's text response.
        input_tokens:  Tokens consumed by the input (prompt).
        output_tokens: Tokens in the response.
        cost_usd:      Estimated cost for this single call.
        model:         Exact model string used for the call.
        raw_response:  Provider-specific raw response object (for debugging).
                       Not serialised to JSON; only available in-process.
    """

    output: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    raw_response: object | None = field(default=None, compare=False)


@runtime_checkable
class Provider(Protocol):
    """Interface that every LLM provider must implement.

    Implement this protocol to add a new provider without subclassing.  The
    :func:`~sivo.providers.registry.get_provider` factory resolves built-in
    providers by name (``"anthropic"``) and custom providers by import path
    (``"my_package.module:MyProvider"``).
    """

    #: Short identifier for this provider, e.g. ``"anthropic"`` or ``"openai"``.
    name: str

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
        """Make an LLM completion call and return a normalised result.

        Args:
            model:         Exact model string (e.g. ``"claude-haiku-4-5"``).
            system_prompt: Optional system/developer prompt.
            messages:      Conversation history as ``[{"role": ..., "content": ...}]``.
            max_tokens:    Maximum tokens to generate.
            timeout:       Per-call timeout in seconds.
            extra_params:  Provider-specific parameters merged into the API call
                           (e.g. ``{"temperature": 0.7}``).

        Returns:
            :class:`CompletionResult` with normalised token counts and cost.
        """
        ...

    def judge(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[dict],
        rubric_name: str,
    ) -> "JudgeVerdict":
        """Make a structured judge call and return a :class:`~sivo.models.JudgeVerdict`.

        The ``system_prompt`` and ``messages`` are pre-formatted by the caller
        (``LLMJudge``). The provider is responsible only for extracting a
        structured verdict using its native mechanism (tool_use, function
        calling, or prompt-based JSON).

        Args:
            model:        Exact model string for the judge.
            system_prompt: Pre-formatted system prompt from the judge.
            messages:     Pre-formatted user message(s) from the judge.
            rubric_name:  Name or description of the rubric (for error context).

        Returns:
            :class:`~sivo.models.JudgeVerdict` with the structured assessment.
        """
        ...
