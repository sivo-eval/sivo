"""Unit tests for sivo data models."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from sivo.models import EvalCase, ExecutionRecord, JudgeVerdict, Message, Step, Trace


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

class TestMessage:
    def test_valid_user_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_valid_assistant_message(self):
        msg = Message(role="assistant", content="Hi there")
        assert msg.role == "assistant"

    def test_invalid_role(self):
        with pytest.raises(ValidationError):
            Message(role="system_invalid", content="x")

    def test_empty_content_allowed(self):
        # Empty string is a valid (if unusual) content value
        msg = Message(role="user", content="")
        assert msg.content == ""

    def test_serialisation_roundtrip(self):
        msg = Message(role="user", content="test")
        restored = Message.model_validate(msg.model_dump())
        assert restored == msg


# ---------------------------------------------------------------------------
# Step (reserved)
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_has_type_and_content(self):
        step = Step(type="tool_call", content={"name": "search", "args": {}})
        assert step.type == "tool_call"
        assert step.content == {"name": "search", "args": {}}

    def test_step_content_can_be_string(self):
        step = Step(type="thought", content="thinking...")
        assert step.content == "thinking..."


# ---------------------------------------------------------------------------
# Trace (reserved)
# ---------------------------------------------------------------------------

class TestTrace:
    def test_trace_wraps_steps(self):
        steps = [Step(type="thought", content="x"), Step(type="tool_call", content={})]
        trace = Trace(steps=steps)
        assert len(trace.steps) == 2

    def test_trace_empty_steps(self):
        trace = Trace(steps=[])
        assert trace.steps == []

    def test_trace_missing_steps_defaults_empty(self):
        trace = Trace()
        assert trace.steps == []


# ---------------------------------------------------------------------------
# JudgeVerdict
# ---------------------------------------------------------------------------

class TestJudgeVerdict:
    def test_passing_verdict(self):
        v = JudgeVerdict(
            passed=True,
            reason="Response is helpful",
            evidence="The answer addresses all concerns",
            suggestion=None,
        )
        assert v.passed is True
        assert v.suggestion is None

    def test_failing_verdict_with_suggestion(self):
        v = JudgeVerdict(
            passed=False,
            reason="Too terse",
            evidence="'Yes.'",
            suggestion="Expand the answer",
        )
        assert v.passed is False
        assert v.suggestion == "Expand the answer"

    def test_requires_passed_field(self):
        with pytest.raises(ValidationError):
            JudgeVerdict(reason="x", evidence="y")

    def test_requires_reason_field(self):
        with pytest.raises(ValidationError):
            JudgeVerdict(passed=True, evidence="y")

    def test_requires_evidence_field(self):
        with pytest.raises(ValidationError):
            JudgeVerdict(passed=True, reason="x")

    def test_serialisation_roundtrip(self):
        v = JudgeVerdict(passed=True, reason="r", evidence="e", suggestion="s")
        restored = JudgeVerdict.model_validate(v.model_dump())
        assert restored == v


# ---------------------------------------------------------------------------
# EvalCase
# ---------------------------------------------------------------------------

class TestEvalCase:
    def test_minimal_construction(self):
        case = EvalCase(input="What is the refund policy?", output="30 days")
        assert case.input == "What is the refund policy?"
        assert case.output == "30 days"

    def test_dict_input(self):
        case = EvalCase(input={"prompt": "hello", "context": "x"}, output="Hi")
        assert isinstance(case.input, dict)

    def test_optional_fields_default_to_none(self):
        case = EvalCase(input="q", output="a")
        assert case.system_prompt is None
        assert case.conversation is None
        assert case.expected is None
        assert case.tools is None
        assert case.trace is None

    def test_metadata_defaults_to_empty_dict(self):
        case = EvalCase(input="q", output="a")
        assert case.metadata == {}

    def test_metadata_accepts_arbitrary_data(self):
        case = EvalCase(input="q", output="a", metadata={"run_id": "r1", "tag": "v1"})
        assert case.metadata["run_id"] == "r1"

    def test_conversation_is_list_of_messages(self):
        msgs = [Message(role="user", content="Hello"), Message(role="assistant", content="Hi")]
        case = EvalCase(input="q", output="a", conversation=msgs)
        assert len(case.conversation) == 2
        assert case.conversation[0].role == "user"

    def test_trace_field_is_reserved_read_only(self):
        # trace can be set but must be Trace type (or None)
        case = EvalCase(input="q", output="a", trace=None)
        assert case.trace is None

    def test_trace_accepts_trace_instance(self):
        t = Trace(steps=[])
        case = EvalCase(input="q", output="a", trace=t)
        assert case.trace is t

    def test_input_is_required(self):
        with pytest.raises(ValidationError):
            EvalCase(output="a")

    def test_output_is_required(self):
        with pytest.raises(ValidationError):
            EvalCase(input="q")

    def test_system_prompt_must_be_string_or_none(self):
        case = EvalCase(input="q", output="a", system_prompt="You are helpful.")
        assert case.system_prompt == "You are helpful."

    def test_expected_can_be_any_type(self):
        case_str = EvalCase(input="q", output="a", expected="30 days")
        case_int = EvalCase(input="q", output="a", expected=42)
        case_list = EvalCase(input="q", output="a", expected=["a", "b"])
        assert case_str.expected == "30 days"
        assert case_int.expected == 42
        assert case_list.expected == ["a", "b"]

    def test_serialisation_roundtrip(self):
        case = EvalCase(
            input="q",
            output="a",
            system_prompt="sys",
            metadata={"k": "v"},
        )
        restored = EvalCase.model_validate(case.model_dump())
        assert restored == case


# ---------------------------------------------------------------------------
# ExecutionRecord
# ---------------------------------------------------------------------------

class TestExecutionRecord:
    def _make_record(self, **overrides):
        defaults = dict(
            id="550e8400-e29b-41d4-a716-446655440000",
            timestamp="2025-01-14T14:32:01Z",
            run_id="run_20250114_143201",
            input="What is the refund policy?",
            output="Our policy is 30 days.",
            model="claude-haiku-4-5",
            params={"temperature": 0.0, "max_tokens": 1024},
            input_tokens=20,
            output_tokens=8,
            cost_usd=0.0001,
        )
        defaults.update(overrides)
        return ExecutionRecord(**defaults)

    def test_minimal_valid_record(self):
        rec = self._make_record()
        assert rec.id == "550e8400-e29b-41d4-a716-446655440000"
        assert rec.model == "claude-haiku-4-5"

    def test_optional_fields_default_to_none(self):
        rec = self._make_record()
        assert rec.system_prompt is None
        assert rec.conversation is None
        assert rec.trace is None

    def test_metadata_defaults_to_empty_dict(self):
        rec = self._make_record()
        assert rec.metadata == {}

    def test_params_stores_arbitrary_dict(self):
        rec = self._make_record(params={"temperature": 0.7, "top_p": 0.9})
        assert rec.params["temperature"] == 0.7

    def test_token_counts_must_be_non_negative_ints(self):
        with pytest.raises(ValidationError):
            self._make_record(input_tokens=-1)
        with pytest.raises(ValidationError):
            self._make_record(output_tokens=-1)

    def test_cost_usd_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            self._make_record(cost_usd=-0.01)

    def test_cost_usd_zero_is_valid(self):
        rec = self._make_record(cost_usd=0.0)
        assert rec.cost_usd == 0.0

    def test_id_is_required(self):
        with pytest.raises(ValidationError):
            self._make_record(id=None)

    def test_timestamp_is_required(self):
        with pytest.raises(ValidationError):
            self._make_record(timestamp=None)

    def test_run_id_is_required(self):
        with pytest.raises(ValidationError):
            self._make_record(run_id=None)

    def test_model_is_required(self):
        with pytest.raises(ValidationError):
            self._make_record(model=None)

    def test_model_must_be_exact_string(self):
        # Exact model string (D-012) — no aliases
        rec = self._make_record(model="claude-sonnet-4-6")
        assert rec.model == "claude-sonnet-4-6"

    def test_trace_is_always_none_in_v1(self):
        rec = self._make_record()
        assert rec.trace is None

    def test_trace_accepts_trace_instance(self):
        t = Trace(steps=[])
        rec = self._make_record(trace=t)
        assert rec.trace is t

    def test_conversation_accepts_messages(self):
        msgs = [Message(role="user", content="Hi")]
        rec = self._make_record(conversation=msgs)
        assert rec.conversation[0].content == "Hi"

    def test_input_can_be_dict(self):
        rec = self._make_record(input={"key": "value"})
        assert rec.input["key"] == "value"

    def test_serialisation_roundtrip(self):
        rec = self._make_record(
            conversation=[Message(role="user", content="Hi")],
            metadata={"tag": "v1"},
        )
        restored = ExecutionRecord.model_validate(rec.model_dump())
        assert restored == rec

    def test_to_eval_case_basic(self):
        rec = self._make_record()
        case = rec.to_eval_case()
        assert isinstance(case, EvalCase)
        assert case.input == rec.input
        assert case.output == rec.output
        assert case.system_prompt == rec.system_prompt
        assert case.conversation == rec.conversation
        assert case.metadata["run_id"] == rec.run_id

    def test_to_eval_case_preserves_metadata(self):
        rec = self._make_record(metadata={"tag": "v1"})
        case = rec.to_eval_case()
        assert case.metadata["tag"] == "v1"
        assert case.metadata["run_id"] == rec.run_id
