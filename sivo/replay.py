"""Replay runner for sivo.

Replays stored :class:`~sivo.models.ExecutionRecord` objects through eval
functions without making any LLM calls. This is the core of sivo's
separation between *execution* (which costs tokens) and *evaluation* (which
is free).

Typical usage::

    sivo replay <run_id> [eval_file.py] [--eval NAME] [--filter KEY=VALUE]
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from sivo.judge import LLMJudge
    from sivo.runner import EvalResult, SessionResult
    from sivo.store import JsonlStore


def parse_filters(filter_strings: list[str]) -> dict[str, str]:
    """Parse a list of ``KEY=VALUE`` strings into a dict.

    Args:
        filter_strings: Strings of the form ``"key=value"``.

    Returns:
        Dict mapping each key to its value.

    Raises:
        ValueError: If any string does not contain ``=``.
    """
    result: dict[str, str] = {}
    for s in filter_strings:
        if "=" not in s:
            raise ValueError(
                f"Invalid filter {s!r}: expected KEY=VALUE format"
            )
        key, _, value = s.partition("=")
        result[key] = value
    return result


def replay_session(
    path: Path,
    *,
    run_id: str,
    store: "JsonlStore | None" = None,
    eval_filter: str | None = None,
    metadata_filter: dict[str, str] | None = None,
    fail_fast: bool = True,
    on_result: "Callable[[EvalResult], None] | None" = None,
    judge: "LLMJudge | None" = None,
) -> "SessionResult":
    """Replay stored records through eval functions without calling the LLM.

    This is a thin wrapper around :func:`~sivo.runner.run_session` that
    makes the intent explicit: no live LLM calls will be made. Records are
    loaded from the JSONL store and each :class:`~sivo.models.EvalCase` is
    pre-populated with the stored ``output``.

    Args:
        path:            File or directory to discover eval functions in.
        run_id:          Run identifier to load records from.
        store:           :class:`~sivo.store.JsonlStore` instance. Defaults
                         to ``./.sivo``.
        eval_filter:     If given, only run the eval function with this exact
                         name.
        metadata_filter: If given, only run against records whose metadata
                         matches all key-value pairs.
        fail_fast:       Stop on the first failure (default ``True``).
        on_result:       Optional callback called after each result.
        judge:           Optional :class:`~sivo.judge.LLMJudge` instance to
                         use as the session-level judge override.

    Returns:
        :class:`~sivo.runner.SessionResult` with all results.
    """
    from sivo.runner import run_session

    return run_session(
        path,
        run_id=run_id,
        store=store,
        eval_filter=eval_filter,
        metadata_filter=metadata_filter,
        fail_fast=fail_fast,
        on_result=on_result,
        judge=judge,
    )
