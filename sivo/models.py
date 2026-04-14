"""Core data models for sivo."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    """A single message in a conversation history."""

    role: Literal["user", "assistant", "system"]
    content: str

    model_config = {"frozen": True}


class Step(BaseModel):
    """A single step in an agent trace. Reserved for future use."""

    type: str
    content: Any

    model_config = {"frozen": True}


class Trace(BaseModel):
    """Agent execution trace. Reserved — not populated in v1."""

    steps: list[Step] = Field(default_factory=list)

    model_config = {"frozen": True}


class JudgeVerdict(BaseModel):
    """Structured output from an LLM judge assessment."""

    passed: bool
    reason: str
    evidence: str
    suggestion: str | None = None

    model_config = {"frozen": True}


class ExecutionInput(BaseModel):
    """Input specification for the :class:`~sivo.runner.ExecutionEngine`.

    Describes a single LLM call to be made. The runner constructs one of these
    per case before calling the execution engine. ``ExecutionRecord`` is
    produced as output; ``EvalCase`` is derived from the record for the eval
    engine.
    """

    input: str | dict
    system_prompt: str | None = None
    conversation: list[Message] | None = None
    expected: Any = None
    metadata: dict = Field(default_factory=dict)
    # Extra model parameters forwarded verbatim (temperature, max_tokens, …)
    params: dict = Field(default_factory=dict)


class EvalCase(BaseModel):
    """Input contract for every eval function.

    Constructed by the runner from an ExecutionRecord. Eval functions must
    retrieve output via ``case.output`` or ``get_response(case)`` — never by
    calling the LLM directly.
    """

    input: str | dict
    output: str
    system_prompt: str | None = None
    conversation: list[Message] | None = None
    expected: Any = None
    metadata: dict = Field(default_factory=dict)
    tools: list | None = None
    # Reserved — read-only; not populated by the runner in v1
    trace: Trace | None = None


class ExecutionRecord(BaseModel):
    """Canonical stored artifact written by the execution engine.

    One record per LLM call. Written to JSONL immediately on completion.
    """

    id: str
    timestamp: str
    run_id: str

    # Input
    input: Any
    system_prompt: str | None = None
    conversation: list[Message] | None = None

    # Output
    output: str

    # Model provenance — exact strings, not aliases (D-012)
    model: str
    params: dict = Field(default_factory=dict)

    # Cost
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)

    # Metadata
    metadata: dict = Field(default_factory=dict)
    # Reserved — always None in v1
    trace: Trace | None = None

    def to_eval_case(self) -> EvalCase:
        """Convert this record into an EvalCase for the eval engine."""
        merged_metadata = {**self.metadata, "run_id": self.run_id}
        return EvalCase(
            input=self.input,
            output=self.output,
            system_prompt=self.system_prompt,
            conversation=self.conversation,
            metadata=merged_metadata,
            trace=self.trace,
        )
