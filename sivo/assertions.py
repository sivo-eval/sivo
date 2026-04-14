"""Deterministic assertion functions for sivo."""

from __future__ import annotations

import re
from typing import Any


class FlakyEvalError(Exception):
    """Raised when an eval yields an indeterminate (flaky) result.

    Eval functions or judge helpers raise this when repeated attempts produce
    split verdicts (e.g. 1-pass / 2-fail across 3 attempts). The runner
    records the outcome as ``FLAKY`` rather than ``FAIL``; it does not affect
    the exit code unless ``--strict-flaky`` is set (Phase 8).
    """


class EvalAssertionError(AssertionError):
    """Assertion failure with structured context.

    Carries the assertion type, a human-readable message, and evidence
    so the terminal UI can render rich failure output.
    """

    def __init__(
        self,
        message: str,
        *,
        assertion_type: str,
        expected: Any = None,
        actual: Any = None,
    ) -> None:
        super().__init__(message)
        self.assertion_type = assertion_type
        self.expected = expected
        self.actual = actual

    def __str__(self) -> str:
        return self.args[0]


def assert_contains(text: str, substring: str) -> None:
    """Assert that *text* contains *substring* (case-sensitive).

    Raises:
        EvalAssertionError: if the substring is not found.
    """
    if substring not in text:
        raise EvalAssertionError(
            f"Expected text to contain {substring!r} but it was not found.\n"
            f"Text: {text!r}",
            assertion_type="assert_contains",
            expected=substring,
            actual=text,
        )


def assert_not_contains(text: str, substring: str) -> None:
    """Assert that *text* does not contain *substring* (case-sensitive).

    Raises:
        EvalAssertionError: if the substring is found.
    """
    if substring in text:
        raise EvalAssertionError(
            f"Expected text NOT to contain {substring!r} but it was found.\n"
            f"Text: {text!r}",
            assertion_type="assert_not_contains",
            expected=f"not {substring!r}",
            actual=text,
        )


def assert_matches_schema(output: Any, schema: Any) -> None:
    """Assert that *output* conforms to *schema*.

    *schema* may be:
    - A Pydantic ``BaseModel`` subclass — validated via ``model_validate``.
    - A ``pydantic.TypeAdapter``-compatible type (e.g. ``dict``, ``list[str]``).
    - A plain ``dict`` — the output must also be a dict with the same keys present.

    For full JSON Schema draft validation, supply a Pydantic model.

    Raises:
        EvalAssertionError: if validation fails.
    """
    import pydantic

    try:
        if isinstance(schema, type) and issubclass(schema, pydantic.BaseModel):
            schema.model_validate(output)
        else:
            adapter = pydantic.TypeAdapter(schema)
            adapter.validate_python(output)
    except pydantic.ValidationError as exc:
        raise EvalAssertionError(
            f"Output does not match schema: {exc}",
            assertion_type="assert_matches_schema",
            expected=schema,
            actual=output,
        ) from exc


def assert_regex(text: str, pattern: str) -> None:
    """Assert that *text* matches the regular expression *pattern*.

    Uses ``re.search`` so the pattern may match anywhere in the text.

    Raises:
        EvalAssertionError: if no match is found.
    """
    if not re.search(pattern, text):
        raise EvalAssertionError(
            f"Text does not match pattern {pattern!r}.\nText: {text!r}",
            assertion_type="assert_regex",
            expected=pattern,
            actual=text,
        )


def assert_length(
    text: str,
    *,
    min: int | None = None,  # noqa: A002
    max: int | None = None,  # noqa: A002
) -> None:
    """Assert that ``len(text)`` is within the specified bounds (inclusive).

    Args:
        text: The string to measure.
        min: Minimum allowed length (inclusive). ``None`` means no lower bound.
        max: Maximum allowed length (inclusive). ``None`` means no upper bound.

    Raises:
        ValueError: if neither *min* nor *max* is specified.
        EvalAssertionError: if the length is out of bounds.
    """
    if min is None and max is None:
        raise ValueError("assert_length requires at least one of min or max.")

    length = len(text)

    if min is not None and length < min:
        raise EvalAssertionError(
            f"Text length {length} is below minimum {min}.\nText: {text!r}",
            assertion_type="assert_length",
            expected=f"length >= {min}",
            actual=length,
        )

    if max is not None and length > max:
        raise EvalAssertionError(
            f"Text length {length} exceeds maximum {max}.\nText: {text!r}",
            assertion_type="assert_length",
            expected=f"length <= {max}",
            actual=length,
        )
