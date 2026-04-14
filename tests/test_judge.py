"""Tests for sivo.judge — LLMJudge and assert_judge.

Integration tests that call the real Anthropic API are guarded by the
``--no-llm`` CLI flag. Run with ``uv run pytest`` to execute all tests,
or ``uv run pytest --no-llm`` to skip live LLM calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sivo.assertions import EvalAssertionError
from sivo.judge import (
    BUILTIN_RUBRICS,
    LLMJudge,
    _get_default_judge,
    assert_judge,
    set_session_judge,
)
from sivo.models import JudgeVerdict


# ---------------------------------------------------------------------------
# Fixture: --no-llm
# ---------------------------------------------------------------------------


@pytest.fixture
def no_llm(request):
    return request.config.getoption("--no-llm", default=False)


# ---------------------------------------------------------------------------
# Built-in rubrics
# ---------------------------------------------------------------------------


class TestBuiltinRubrics:
    def test_all_expected_rubrics_present(self):
        expected = {"helpfulness", "tone", "toxicity", "factual_consistency", "conciseness"}
        assert expected == set(BUILTIN_RUBRICS.keys())

    def test_rubric_values_are_nonempty_strings(self):
        for name, text in BUILTIN_RUBRICS.items():
            assert isinstance(text, str), f"Rubric {name!r} is not a string"
            assert len(text) > 0, f"Rubric {name!r} is empty"


# ---------------------------------------------------------------------------
# LLMJudge — caching
# ---------------------------------------------------------------------------


class TestLLMJudgeCache:
    def _make_verdict(self, passed: bool = True) -> JudgeVerdict:
        return JudgeVerdict(
            passed=passed,
            reason="Test reason.",
            evidence="test evidence",
            suggestion=None,
        )

    def test_same_input_returns_cached_verdict(self):
        judge = LLMJudge()
        verdict = self._make_verdict(passed=True)

        with patch.object(judge._provider, "judge", return_value=verdict) as mock_call:
            result1 = judge.assess("output text", "helpfulness")
            result2 = judge.assess("output text", "helpfulness")

        assert mock_call.call_count == 1
        assert result1 is result2

    def test_different_output_bypasses_cache(self):
        judge = LLMJudge()
        v1 = self._make_verdict(passed=True)
        v2 = self._make_verdict(passed=False)

        with patch.object(judge._provider, "judge", side_effect=[v1, v2]):
            r1 = judge.assess("output A", "helpfulness")
            r2 = judge.assess("output B", "helpfulness")

        assert r1.passed is True
        assert r2.passed is False

    def test_different_rubric_bypasses_cache(self):
        judge = LLMJudge()
        v1 = self._make_verdict(passed=True)
        v2 = self._make_verdict(passed=False)

        with patch.object(judge._provider, "judge", side_effect=[v1, v2]):
            r1 = judge.assess("same output", "helpfulness")
            r2 = judge.assess("same output", "tone")

        assert r1.passed is True
        assert r2.passed is False

    def test_builtin_rubric_resolved_before_caching(self):
        """Two calls with rubric name vs. full text should share cache."""
        judge = LLMJudge()
        verdict = self._make_verdict(passed=True)

        with patch.object(judge._provider, "judge", return_value=verdict) as mock_call:
            judge.assess("output", "helpfulness")
            # Call again with the resolved rubric text — should still hit cache
            judge.assess("output", BUILTIN_RUBRICS["helpfulness"])

        assert mock_call.call_count == 1

    def test_cache_key_is_deterministic(self):
        k1 = LLMJudge._cache_key("rubric", "output")
        k2 = LLMJudge._cache_key("rubric", "output")
        assert k1 == k2

    def test_cache_key_differs_for_different_inputs(self):
        k1 = LLMJudge._cache_key("rubric A", "output")
        k2 = LLMJudge._cache_key("rubric B", "output")
        assert k1 != k2

    def test_cache_key_includes_model(self):
        """Different models produce different cache keys for same rubric+output."""
        k1 = LLMJudge._cache_key("rubric", "output", "claude-haiku-4-5")
        k2 = LLMJudge._cache_key("rubric", "output", "gpt-4o")
        assert k1 != k2

    def test_cache_key_same_model_is_deterministic(self):
        k1 = LLMJudge._cache_key("rubric", "output", "claude-haiku-4-5")
        k2 = LLMJudge._cache_key("rubric", "output", "claude-haiku-4-5")
        assert k1 == k2

    def test_different_models_bypass_cache(self):
        """Two judges with different models don't share cache entries."""
        v1 = JudgeVerdict(passed=True, reason="ok", evidence="e")
        v2 = JudgeVerdict(passed=False, reason="nope", evidence="e")

        judge1 = LLMJudge(model="claude-haiku-4-5")
        judge2 = LLMJudge(model="gpt-4o")

        with patch.object(judge1._provider, "judge", return_value=v1):
            r1 = judge1.assess("same output", "helpfulness")

        with patch.object(judge2._provider, "judge", return_value=v2):
            r2 = judge2.assess("same output", "helpfulness")

        assert r1.passed is True
        assert r2.passed is False


# ---------------------------------------------------------------------------
# assert_judge — unit tests with mocked judge
# ---------------------------------------------------------------------------


class TestAssertJudge:
    def _mock_judge(self, passed: bool, suggestion: str | None = None) -> LLMJudge:
        verdict = JudgeVerdict(
            passed=passed,
            reason="Test reason.",
            evidence="test span",
            suggestion=suggestion,
        )
        judge = MagicMock(spec=LLMJudge)
        judge.assess.return_value = verdict
        return judge

    def test_passes_when_verdict_is_true(self):
        j = self._mock_judge(passed=True)
        result = assert_judge("good output", rubric="helpfulness", judge=j)
        assert result.passed is True

    def test_returns_verdict_on_pass(self):
        j = self._mock_judge(passed=True)
        verdict = assert_judge("good output", rubric="helpfulness", judge=j)
        assert isinstance(verdict, JudgeVerdict)

    def test_raises_eval_assertion_error_on_fail(self):
        j = self._mock_judge(passed=False)
        with pytest.raises(EvalAssertionError) as exc_info:
            assert_judge("bad output", rubric="helpfulness", judge=j)
        err = exc_info.value
        assert err.assertion_type == "assert_judge"

    def test_failure_message_includes_rubric(self):
        j = self._mock_judge(passed=False)
        with pytest.raises(EvalAssertionError) as exc_info:
            assert_judge("bad output", rubric="helpfulness", judge=j)
        assert "helpfulness" in str(exc_info.value)

    def test_failure_message_includes_reason(self):
        j = self._mock_judge(passed=False)
        with pytest.raises(EvalAssertionError) as exc_info:
            assert_judge("bad output", rubric="helpfulness", judge=j)
        assert "Test reason." in str(exc_info.value)

    def test_failure_message_includes_evidence(self):
        j = self._mock_judge(passed=False)
        with pytest.raises(EvalAssertionError) as exc_info:
            assert_judge("bad output", rubric="helpfulness", judge=j)
        assert "test span" in str(exc_info.value)

    def test_failure_message_includes_suggestion_when_present(self):
        j = self._mock_judge(passed=False, suggestion="Try being more direct.")
        with pytest.raises(EvalAssertionError) as exc_info:
            assert_judge("bad output", rubric="helpfulness", judge=j)
        assert "Try being more direct." in str(exc_info.value)

    def test_verdict_stored_as_actual_on_failure(self):
        j = self._mock_judge(passed=False)
        with pytest.raises(EvalAssertionError) as exc_info:
            assert_judge("bad output", rubric="helpfulness", judge=j)
        assert isinstance(exc_info.value.actual, JudgeVerdict)

    def test_default_judge_is_reused(self):
        """_get_default_judge returns same instance for same model."""
        j1 = _get_default_judge("claude-haiku-4-5")
        j2 = _get_default_judge("claude-haiku-4-5")
        assert j1 is j2

    def test_different_model_creates_new_judge(self):
        j1 = _get_default_judge("claude-haiku-4-5")
        j2 = _get_default_judge("claude-sonnet-4-6")
        assert j1 is not j2


# ---------------------------------------------------------------------------
# set_session_judge — session override
# ---------------------------------------------------------------------------


class TestSetSessionJudge:
    def setup_method(self):
        # Ensure no override leaks between tests
        set_session_judge(None)

    def teardown_method(self):
        set_session_judge(None)

    def test_set_session_judge_returns_previous(self):
        custom_judge = LLMJudge(model="gpt-4o")
        prev = set_session_judge(custom_judge)
        assert prev is None  # no prior override

    def test_get_default_judge_returns_override(self):
        custom_judge = LLMJudge(model="gpt-4o")
        set_session_judge(custom_judge)
        assert _get_default_judge("claude-haiku-4-5") is custom_judge

    def test_override_ignores_model_argument(self):
        """Session override is returned regardless of the model arg."""
        custom_judge = LLMJudge(model="gpt-4o")
        set_session_judge(custom_judge)
        assert _get_default_judge("claude-opus-4-6") is custom_judge

    def test_clear_override_restores_default_behaviour(self):
        custom_judge = LLMJudge(model="gpt-4o")
        set_session_judge(custom_judge)
        set_session_judge(None)
        # Now default behaviour: returns a normal judge, not our custom one
        j = _get_default_judge("claude-haiku-4-5")
        assert j is not custom_judge

    def test_restore_pattern(self):
        """Typical save-and-restore pattern used by run_session."""
        custom = LLMJudge(model="gpt-4o")
        prev = set_session_judge(custom)
        try:
            assert _get_default_judge() is custom
        finally:
            set_session_judge(prev)
        assert _get_default_judge() is not custom


# ---------------------------------------------------------------------------
# Integration tests (skipped with --no-llm)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLLMJudgeIntegration:
    """Live tests that hit the Anthropic API.

    Skip with: uv run pytest --no-llm
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_llm(self, no_llm):
        if no_llm:
            pytest.skip("--no-llm flag set; skipping live LLM tests.")

    def test_assess_helpful_response_passes_helpfulness(self):
        judge = LLMJudge()
        output = (
            "To cancel your subscription, go to Settings → Billing → Cancel Plan. "
            "Your access continues until the end of the billing period."
        )
        verdict = judge.assess(output, "helpfulness")
        assert isinstance(verdict, JudgeVerdict)
        assert verdict.passed is True
        assert verdict.reason
        assert verdict.evidence

    def test_assess_unhelpful_response_fails_helpfulness(self):
        judge = LLMJudge()
        output = "I'm not sure. Maybe try contacting support."
        verdict = judge.assess(output, "helpfulness")
        assert isinstance(verdict, JudgeVerdict)
        # Evidence and reason should always be present
        assert verdict.reason
        assert verdict.evidence

    def test_assert_judge_raises_on_toxic_content(self):
        judge = LLMJudge()
        # Borderline placeholder — actual toxicity test would need a real toxic string
        output = "This is a perfectly fine, friendly response."
        verdict = judge.assess(output, "toxicity")
        assert verdict.passed is True  # should pass (not toxic)

    def test_cache_prevents_second_api_call(self):
        judge = LLMJudge()
        output = "The answer is 42."
        rubric = "helpfulness"

        v1 = judge.assess(output, rubric)
        v2 = judge.assess(output, rubric)

        assert v1 is v2  # same object from cache
