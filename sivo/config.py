"""sivo.toml configuration loader.

Reads project-level settings from ``sivo.toml`` at the project root (or
any ancestor directory). Returns typed defaults when no file is present.

Usage::

    from sivo.config import load_config

    cfg = load_config()           # searches from cwd upward
    print(cfg.store_path)         # ".sivo" by default
    print(cfg.default_model)      # "claude-haiku-4-5" by default
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SivoConfig:
    """Typed configuration loaded from ``sivo.toml``.

    All fields carry sensible defaults so the file is entirely optional.

    Attributes:
        default_model:         LLM model used by :class:`~sivo.runner.ExecutionEngine`.
        concurrency:           Max simultaneous LLM calls.
        timeout:               Per-call timeout in seconds.
        store_path:            Root directory for the JSONL record store.
        provider:              LLM provider to use for execution calls.
                               Built-in values: ``"anthropic"`` (default).
                               Custom providers: ``"my_package.module:ClassName"``.
        judge_model:           LLM model used by the judge (``assert_judge``).
        judge_provider:        Provider to use for judge calls.  Defaults to the
                               same as *provider* when empty.
        judge_retry_attempts:  Number of judge attempts per assertion (3 = majority vote).
        cost_warn_above_usd:   Emit a warning if session cost exceeds this threshold.
    """

    # [sivo]
    default_model: str = "claude-haiku-4-5"
    concurrency: int = 10
    timeout: float = 30.0
    store_path: str = ".sivo"
    provider: str = "anthropic"

    # [sivo.judge]
    judge_model: str = "claude-haiku-4-5"
    judge_provider: str = ""  # empty = use same provider as execution
    judge_retry_attempts: int = 1

    # [sivo.cost]
    cost_warn_above_usd: float = 1.00


def load_config(search_path: Path | None = None) -> SivoConfig:
    """Load ``sivo.toml`` and return an :class:`SivoConfig`.

    Searches *search_path* (defaults to the current working directory) and its
    ancestor directories for an ``sivo.toml`` file.  Returns all-default
    config if no file is found — the file is optional.

    Args:
        search_path: Directory to start searching from.  Defaults to
                     ``Path.cwd()``.

    Returns:
        :class:`SivoConfig` with values from the first ``sivo.toml``
        found, or all defaults if none is present.
    """
    config_file = _find_config(search_path or Path.cwd())
    if config_file is None:
        return SivoConfig()

    with open(config_file, "rb") as fh:
        data = tomllib.load(fh)

    return _parse_config(data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_config(start: Path) -> Path | None:
    """Walk upward from *start* looking for ``sivo.toml``.

    Args:
        start: Directory to begin the search (should be an absolute path).

    Returns:
        Absolute :class:`~pathlib.Path` of the first ``sivo.toml`` found,
        or ``None`` if none exists on the path to the filesystem root.
    """
    current = start.resolve()
    for directory in [current, *current.parents]:
        candidate = directory / "sivo.toml"
        if candidate.is_file():
            return candidate
    return None


def _parse_config(data: dict) -> SivoConfig:
    """Parse a raw TOML mapping into an :class:`SivoConfig`.

    Unknown keys are silently ignored so that forward-compatible ``sivo.toml``
    files do not break older versions of the library.

    Args:
        data: Top-level TOML dict as returned by ``tomllib.load``.

    Returns:
        Populated :class:`SivoConfig`.
    """
    cfg = SivoConfig()

    top: dict = data.get("sivo", {})

    if "default_model" in top:
        cfg.default_model = str(top["default_model"])
    if "concurrency" in top:
        cfg.concurrency = int(top["concurrency"])
    if "timeout" in top:
        cfg.timeout = float(top["timeout"])
    if "store_path" in top:
        cfg.store_path = str(top["store_path"])
    if "provider" in top:
        cfg.provider = str(top["provider"])

    judge: dict = top.get("judge", {})
    if "default_model" in judge:
        cfg.judge_model = str(judge["default_model"])
    if "provider" in judge:
        cfg.judge_provider = str(judge["provider"])
    if "retry_attempts" in judge:
        cfg.judge_retry_attempts = int(judge["retry_attempts"])

    cost: dict = top.get("cost", {})
    if "warn_above_usd" in cost:
        cfg.cost_warn_above_usd = float(cost["warn_above_usd"])

    return cfg
