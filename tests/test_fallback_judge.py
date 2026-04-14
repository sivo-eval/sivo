"""Tests for sivo.providers._fallback_judge — prompt-based JSON judge utility."""

from __future__ import annotations

import json

import pytest

from sivo.models import JudgeVerdict
from sivo.providers._fallback_judge import (
    _VERDICT_SCHEMA_BLOCK,
    build_fallback_system_prompt,
    parse_fallback_response,
)


# ---------------------------------------------------------------------------
# build_fallback_system_prompt
# ---------------------------------------------------------------------------


def test_build_fallback_system_prompt_appends_schema():
    base = "You are a judge."
    result = build_fallback_system_prompt(base)
    assert result.startswith("You are a judge.")
    assert "passed" in result
    assert "reason" in result
    assert "evidence" in result


def test_build_fallback_system_prompt_strips_trailing_whitespace():
    base = "You are a judge.   "
    result = build_fallback_system_prompt(base)
    # The base should be stripped before appending the schema
    assert result.startswith("You are a judge.")


def test_build_fallback_system_prompt_nonempty():
    result = build_fallback_system_prompt("base")
    assert len(result) > len("base")


# ---------------------------------------------------------------------------
# parse_fallback_response — happy paths
# ---------------------------------------------------------------------------


def test_parse_fallback_response_basic():
    raw = json.dumps({
        "passed": True,
        "reason": "Looks good.",
        "evidence": "some span",
        "suggestion": None,
    })
    verdict = parse_fallback_response(raw)
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.passed is True
    assert verdict.reason == "Looks good."
    assert verdict.evidence == "some span"
    assert verdict.suggestion is None


def test_parse_fallback_response_failed_verdict():
    raw = json.dumps({
        "passed": False,
        "reason": "Fails rubric.",
        "evidence": "bad span",
        "suggestion": "Try being clearer.",
    })
    verdict = parse_fallback_response(raw)
    assert verdict.passed is False
    assert verdict.suggestion == "Try being clearer."


def test_parse_fallback_response_strips_markdown_fences():
    raw = "```json\n{\"passed\": true, \"reason\": \"ok\", \"evidence\": \"e\"}\n```"
    verdict = parse_fallback_response(raw)
    assert verdict.passed is True


def test_parse_fallback_response_strips_plain_fences():
    raw = "```\n{\"passed\": false, \"reason\": \"nope\", \"evidence\": \"x\"}\n```"
    verdict = parse_fallback_response(raw)
    assert verdict.passed is False


def test_parse_fallback_response_extra_whitespace():
    raw = "  \n  " + json.dumps({
        "passed": True,
        "reason": "fine",
        "evidence": "e",
    }) + "\n  "
    verdict = parse_fallback_response(raw)
    assert verdict.passed is True


# ---------------------------------------------------------------------------
# parse_fallback_response — error cases
# ---------------------------------------------------------------------------


def test_parse_fallback_response_invalid_json_raises():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_fallback_response("this is not json")


def test_parse_fallback_response_missing_passed_raises():
    raw = json.dumps({"reason": "ok", "evidence": "e"})
    with pytest.raises(ValueError, match="missing fields"):
        parse_fallback_response(raw)


def test_parse_fallback_response_missing_reason_raises():
    raw = json.dumps({"passed": True, "evidence": "e"})
    with pytest.raises(ValueError, match="missing fields"):
        parse_fallback_response(raw)


def test_parse_fallback_response_missing_evidence_raises():
    raw = json.dumps({"passed": True, "reason": "ok"})
    with pytest.raises(ValueError, match="missing fields"):
        parse_fallback_response(raw)


def test_parse_fallback_response_rubric_name_in_error():
    with pytest.raises(ValueError, match="my-rubric"):
        parse_fallback_response("not json", rubric_name="my-rubric")
