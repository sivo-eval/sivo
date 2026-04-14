"""Tests for the sivo assertion library."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from sivo.assertions import (
    EvalAssertionError,
    assert_contains,
    assert_length,
    assert_matches_schema,
    assert_not_contains,
    assert_regex,
)


# ---------------------------------------------------------------------------
# EvalAssertionError
# ---------------------------------------------------------------------------


class TestEvalAssertionError:
    def test_is_assertion_error_subclass(self):
        err = EvalAssertionError("msg", assertion_type="assert_contains")
        assert isinstance(err, AssertionError)

    def test_carries_assertion_type(self):
        err = EvalAssertionError("msg", assertion_type="assert_regex")
        assert err.assertion_type == "assert_regex"

    def test_carries_expected_and_actual(self):
        err = EvalAssertionError(
            "msg", assertion_type="assert_contains", expected="foo", actual="bar"
        )
        assert err.expected == "foo"
        assert err.actual == "bar"

    def test_str_returns_message(self):
        err = EvalAssertionError("failure message", assertion_type="x")
        assert str(err) == "failure message"

    def test_defaults_expected_actual_to_none(self):
        err = EvalAssertionError("msg", assertion_type="x")
        assert err.expected is None
        assert err.actual is None


# ---------------------------------------------------------------------------
# assert_contains
# ---------------------------------------------------------------------------


class TestAssertContains:
    def test_passes_when_substring_present(self):
        assert_contains("hello world", "world")  # should not raise

    def test_passes_with_exact_match(self):
        assert_contains("exact", "exact")

    def test_fails_when_substring_absent(self):
        with pytest.raises(EvalAssertionError) as exc_info:
            assert_contains("hello world", "python")
        err = exc_info.value
        assert err.assertion_type == "assert_contains"
        assert err.expected == "python"
        assert "python" in str(err)

    def test_case_sensitive_fails(self):
        with pytest.raises(EvalAssertionError):
            assert_contains("Hello World", "hello")

    def test_case_sensitive_passes(self):
        assert_contains("Hello World", "Hello")

    def test_empty_substring_always_passes(self):
        assert_contains("anything", "")

    def test_empty_text_fails_nonempty_substring(self):
        with pytest.raises(EvalAssertionError):
            assert_contains("", "x")


# ---------------------------------------------------------------------------
# assert_not_contains
# ---------------------------------------------------------------------------


class TestAssertNotContains:
    def test_passes_when_absent(self):
        assert_not_contains("hello world", "python")

    def test_fails_when_present(self):
        with pytest.raises(EvalAssertionError) as exc_info:
            assert_not_contains("hello world", "world")
        err = exc_info.value
        assert err.assertion_type == "assert_not_contains"
        assert "world" in str(err)

    def test_case_sensitive_passes_wrong_case(self):
        assert_not_contains("Hello World", "hello")  # different case → absent

    def test_empty_text_passes_nonempty_substring(self):
        assert_not_contains("", "x")

    def test_empty_substring_always_fails(self):
        # Empty string is always "in" any string
        with pytest.raises(EvalAssertionError):
            assert_not_contains("anything", "")


# ---------------------------------------------------------------------------
# assert_matches_schema
# ---------------------------------------------------------------------------


class SampleModel(BaseModel):
    name: str
    age: int


class TestAssertMatchesSchema:
    def test_passes_valid_pydantic_model(self):
        assert_matches_schema({"name": "Alice", "age": 30}, SampleModel)

    def test_fails_invalid_pydantic_model_missing_field(self):
        with pytest.raises(EvalAssertionError) as exc_info:
            assert_matches_schema({"name": "Alice"}, SampleModel)
        err = exc_info.value
        assert err.assertion_type == "assert_matches_schema"

    def test_fails_invalid_pydantic_model_wrong_type(self):
        with pytest.raises(EvalAssertionError):
            assert_matches_schema({"name": "Alice", "age": "not_an_int"}, SampleModel)

    def test_passes_dict_type_annotation(self):
        assert_matches_schema({"key": "value"}, dict)

    def test_passes_list_of_str(self):
        assert_matches_schema(["a", "b", "c"], list[str])

    def test_fails_list_of_str_with_wrong_element(self):
        with pytest.raises(EvalAssertionError):
            assert_matches_schema(["a", 1, "c"], list[str])

    def test_actual_set_on_failure(self):
        bad_data = {"name": "Alice"}
        with pytest.raises(EvalAssertionError) as exc_info:
            assert_matches_schema(bad_data, SampleModel)
        assert exc_info.value.actual is bad_data


# ---------------------------------------------------------------------------
# assert_regex
# ---------------------------------------------------------------------------


class TestAssertRegex:
    def test_passes_simple_match(self):
        assert_regex("hello world", r"world")

    def test_passes_partial_match(self):
        assert_regex("The price is $12.50", r"\$[\d.]+")

    def test_passes_anchored_match(self):
        assert_regex("hello", r"^hello$")

    def test_fails_no_match(self):
        with pytest.raises(EvalAssertionError) as exc_info:
            assert_regex("hello world", r"\d+")
        err = exc_info.value
        assert err.assertion_type == "assert_regex"
        assert r"\d+" in str(err)

    def test_fails_wrong_anchor(self):
        with pytest.raises(EvalAssertionError):
            assert_regex("hello world", r"^world")

    def test_pattern_stored_as_expected(self):
        with pytest.raises(EvalAssertionError) as exc_info:
            assert_regex("abc", r"\d+")
        assert exc_info.value.expected == r"\d+"

    def test_actual_is_text(self):
        with pytest.raises(EvalAssertionError) as exc_info:
            assert_regex("abc", r"\d+")
        assert exc_info.value.actual == "abc"


# ---------------------------------------------------------------------------
# assert_length
# ---------------------------------------------------------------------------


class TestAssertLength:
    def test_passes_within_bounds(self):
        assert_length("hello", min=1, max=10)

    def test_passes_exact_min(self):
        assert_length("hello", min=5)

    def test_passes_exact_max(self):
        assert_length("hello", max=5)

    def test_fails_below_min(self):
        with pytest.raises(EvalAssertionError) as exc_info:
            assert_length("hi", min=5)
        err = exc_info.value
        assert err.assertion_type == "assert_length"
        assert err.actual == 2

    def test_fails_above_max(self):
        with pytest.raises(EvalAssertionError) as exc_info:
            assert_length("hello world", max=5)
        err = exc_info.value
        assert err.assertion_type == "assert_length"
        assert err.actual == 11

    def test_requires_at_least_one_bound(self):
        with pytest.raises(ValueError, match="at least one"):
            assert_length("hello")

    def test_empty_string_with_max(self):
        assert_length("", max=10)

    def test_empty_string_fails_min(self):
        with pytest.raises(EvalAssertionError):
            assert_length("", min=1)

    def test_min_equals_max_passes_exact(self):
        assert_length("hello", min=5, max=5)

    def test_min_equals_max_fails(self):
        with pytest.raises(EvalAssertionError):
            assert_length("hi", min=5, max=5)
