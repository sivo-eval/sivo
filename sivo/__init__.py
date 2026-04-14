"""Sivo — developer-first LLM evaluation library.

Quick start::

    import sivo
    from sivo.assertions import assert_contains, assert_judge

    def eval_my_feature(case):
        response = case.output
        assert_contains(response, "expected text")
        assert_judge(response, rubric="helpfulness")

Run with ``sivo run`` (replay against stored records) or use
:func:`~sivo.runner.run_session` programmatically.
"""

from sivo.assertions import (
    EvalAssertionError,
    FlakyEvalError,
    assert_contains,
    assert_length,
    assert_matches_schema,
    assert_not_contains,
    assert_regex,
)
from sivo.config import SivoConfig, load_config
from sivo.fixtures import fixture
from sivo.judge import assert_judge
from sivo.models import EvalCase, ExecutionRecord, JudgeVerdict
from sivo.runner import EvalResult, SessionResult, get_response

__all__ = [
    # Core eval primitives
    "fixture",
    "get_response",
    # Models
    "EvalCase",
    "ExecutionRecord",
    "JudgeVerdict",
    # Results
    "EvalResult",
    "SessionResult",
    # Assertions
    "assert_contains",
    "assert_not_contains",
    "assert_regex",
    "assert_length",
    "assert_matches_schema",
    "assert_judge",
    "EvalAssertionError",
    "FlakyEvalError",
    # Config
    "SivoConfig",
    "load_config",
]
