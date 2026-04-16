"""sivo CLI entry point.

Invoked as ``sivo <command>`` (via the ``[project.scripts]`` entry point)
or ``python -m sivo.cli <command>``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sivo.config import SivoConfig
    from sivo.judge import LLMJudge


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------


def _resolve_judge(args: argparse.Namespace, config: "SivoConfig") -> "LLMJudge | None":
    """Create a custom :class:`~sivo.judge.LLMJudge` from CLI args + config.

    Returns ``None`` when the default AnthropicProvider + default model should
    be used (avoids an unnecessary import of ``sivo.judge`` on every run).

    The precedence for each setting is: CLI flag > sivo.toml > built-in default.
    """
    from sivo.judge import LLMJudge

    # Effective execution provider (used as fallback for judge provider)
    exec_provider = (
        getattr(args, "provider", None) or config.provider or "anthropic"
    )
    # Effective judge provider — falls back to execution provider when unset
    judge_provider_name = (
        getattr(args, "judge_provider", None)
        or (config.judge_provider if config.judge_provider else None)
        or exec_provider
    )
    # Effective judge model
    judge_model = getattr(args, "judge_model", None) or config.judge_model

    # No-op when both settings match the built-in defaults
    if judge_provider_name == "anthropic" and judge_model == LLMJudge.DEFAULT_MODEL:
        return None

    from sivo.providers.registry import get_provider

    try:
        provider = get_provider(judge_provider_name)
    except (ValueError, ImportError, TypeError) as exc:
        raise ValueError(
            f"Cannot load judge provider {judge_provider_name!r}: {exc}"
        ) from exc

    return LLMJudge(model=judge_model, provider=provider)


def _cmd_replay(args: argparse.Namespace) -> int:
    """Execute ``sivo replay``.

    Returns the process exit code (0 = all pass, 1 = any fail, 2 = error).
    """
    from sivo import report
    from sivo.config import SivoConfig, load_config
    from sivo.replay import parse_filters, replay_session
    from sivo.runner import EvalResult
    from sivo.store import JsonlStore

    config = load_config()
    console = report.make_console()

    path = Path(args.path)
    if not path.exists():
        _err(f"Path does not exist: {path}")
        return 2

    store_path = args.store_path or config.store_path
    store = JsonlStore(Path(store_path))

    # Parse --filter KEY=VALUE arguments
    try:
        metadata_filter = parse_filters(args.filter) if args.filter else None
    except ValueError as exc:
        _err(str(exc))
        return 2

    # Resolve judge provider from args + config
    try:
        judge = _resolve_judge(args, config)
    except ValueError as exc:
        _err(str(exc))
        return 2

    verbose: int = args.verbose

    def _on_result(result: EvalResult) -> None:
        report.print_result(result, verbose=verbose, console=console)

    try:
        session = replay_session(
            path,
            run_id=args.run_id,
            store=store,
            eval_filter=args.eval,
            metadata_filter=metadata_filter,
            fail_fast=not args.no_fail_fast,
            on_result=_on_result,
            judge=judge,
        )
    except ValueError as exc:
        _err(str(exc))
        return 2

    strict_flaky: bool = getattr(args, "strict_flaky", False)

    report.print_receipt(session, console=console)
    report.print_cost_warning(session, config.cost_warn_above_usd, console=console)
    report.write_json_summary(session, Path(store_path))
    if getattr(args, "junit_xml", None):
        report.write_junit_xml(session, Path(args.junit_xml), strict_flaky=strict_flaky)

    return 0 if session.is_success(strict_flaky=strict_flaky) else 1


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute ``sivo run``.

    Returns the process exit code (0 = all pass, 1 = any fail, 2 = error).
    """
    from sivo import report
    from sivo.config import SivoConfig, load_config
    from sivo.runner import EvalResult, run_session
    from sivo.store import JsonlStore

    config = load_config()
    console = report.make_console()

    path = Path(args.path)
    if not path.exists():
        _err(f"Path does not exist: {path}")
        return 2

    store_path = args.store_path or config.store_path
    store = JsonlStore(Path(store_path))

    verbose: int = args.verbose

    # Stream each result to the console as it completes (progress display).
    def _on_result(result: EvalResult) -> None:
        report.print_result(result, verbose=verbose, console=console)

    # --pdb-llm: create a REPL hook that pauses on each failure
    pdb_hook = None
    if getattr(args, "pdb_llm", False):
        from sivo.providers.registry import get_provider
        from sivo.repl import make_pdb_hook
        exec_provider_name = getattr(args, "provider", None) or config.provider
        try:
            exec_provider = get_provider(exec_provider_name)
        except (ValueError, ImportError, TypeError) as exc:
            _err(f"Cannot load provider {exec_provider_name!r}: {exc}")
            return 2
        pdb_hook = make_pdb_hook(
            console=console,
            provider=exec_provider,
            model=config.default_model,
        )

    # Resolve judge provider from args + config
    try:
        judge = _resolve_judge(args, config)
    except ValueError as exc:
        _err(str(exc))
        return 2

    try:
        session = run_session(
            path,
            run_id=args.run_id,
            store=store,
            eval_filter=args.eval,
            fail_fast=not args.no_fail_fast,
            on_result=_on_result,
            pdb_hook=pdb_hook,
            judge=judge,
        )
    except ValueError as exc:
        _err(str(exc))
        return 2

    strict_flaky: bool = getattr(args, "strict_flaky", False)

    report.print_receipt(session, console=console)
    report.print_cost_warning(session, config.cost_warn_above_usd, console=console)
    report.write_json_summary(session, Path(store_path))
    if getattr(args, "junit_xml", None):
        report.write_junit_xml(session, Path(args.junit_xml), strict_flaky=strict_flaky)

    return 0 if session.is_success(strict_flaky=strict_flaky) else 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _err(msg: str) -> None:
    print(f"sivo: error: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sivo",
        description="Developer-first LLM evaluation tool.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # --- run ---
    run_p = sub.add_parser(
        "run",
        help="Discover and run eval functions.",
        description=(
            "Discover eval_* functions in PATH and run them against stored "
            "ExecutionRecords. Use --run-id to specify which JSONL run to load."
        ),
    )
    run_p.add_argument(
        "path",
        nargs="?",
        default=".",
        help="File or directory to discover eval functions in (default: current directory).",
    )
    run_p.add_argument(
        "--run-id",
        metavar="RUN_ID",
        default=None,
        help="Load ExecutionRecords from this run ID (stored in .sivo/records/).",
    )
    run_p.add_argument(
        "--eval",
        metavar="NAME",
        default=None,
        help="Run only the eval function with this exact name.",
    )
    run_p.add_argument(
        "--no-fail-fast",
        action="store_true",
        default=False,
        help=(
            "Continue running all evals even after a failure "
            "(default: stop on first failure)."
        ),
    )
    run_p.add_argument(
        "--store-path",
        metavar="PATH",
        default=None,
        help="Root directory of the sivo data store (default: .sivo or sivo.toml store_path).",
    )
    run_p.add_argument(
        "-v",
        dest="verbose",
        action="count",
        default=0,
        help="Increase output verbosity (-v for evidence, -vv for full JSON).",
    )
    run_p.add_argument(
        "--pdb-llm",
        action="store_true",
        default=False,
        dest="pdb_llm",
        help=(
            "Drop into an interactive REPL on every failing eval. "
            "Allows inspecting context, hot-swapping system_prompt, and retrying."
        ),
    )
    run_p.add_argument(
        "--junit-xml",
        metavar="PATH",
        default=None,
        dest="junit_xml",
        help="Write a JUnit-compatible XML report to PATH.",
    )
    run_p.add_argument(
        "--strict-flaky",
        action="store_true",
        default=False,
        dest="strict_flaky",
        help="Treat FLAKY results as failures (exit code 1, fail in JUnit XML).",
    )
    run_p.add_argument(
        "--provider",
        metavar="PROVIDER",
        default=None,
        help=(
            "LLM provider to use for execution. "
            "Built-ins: 'anthropic' (default), 'openai'. "
            "Custom: 'my_package.module:MyProvider'."
        ),
    )
    run_p.add_argument(
        "--judge-provider",
        metavar="PROVIDER",
        default=None,
        dest="judge_provider",
        help=(
            "LLM provider for the LLM judge. "
            "Defaults to --provider (or 'anthropic' if unset). "
            "Built-ins: 'anthropic', 'openai'. "
            "Custom: 'my_package.module:MyProvider'."
        ),
    )
    run_p.add_argument(
        "--judge-model",
        metavar="MODEL",
        default=None,
        dest="judge_model",
        help=(
            "Model to use for the LLM judge "
            "(default: from sivo.toml or 'claude-haiku-4-5')."
        ),
    )

    # --- replay ---
    replay_p = sub.add_parser(
        "replay",
        help="Replay stored records through eval functions (no LLM calls).",
        description=(
            "Load ExecutionRecords for RUN_ID from the store and run eval "
            "functions against them without making any LLM calls. "
            "Use --filter KEY=VALUE to narrow which records are replayed."
        ),
    )
    replay_p.add_argument(
        "run_id",
        metavar="RUN_ID",
        help="Run ID to load records from (stored in .sivo/records/).",
    )
    replay_p.add_argument(
        "path",
        nargs="?",
        default=".",
        help="File or directory to discover eval functions in (default: current directory).",
    )
    replay_p.add_argument(
        "--eval",
        metavar="NAME",
        default=None,
        help="Run only the eval function with this exact name.",
    )
    replay_p.add_argument(
        "--filter",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        dest="filter",
        help="Filter records by metadata field (repeatable, e.g. --filter env=prod).",
    )
    replay_p.add_argument(
        "--no-fail-fast",
        action="store_true",
        default=False,
        help="Continue running all evals even after a failure.",
    )
    replay_p.add_argument(
        "--store-path",
        metavar="PATH",
        default=None,
        help="Root directory of the sivo data store (default: .sivo or sivo.toml store_path).",
    )
    replay_p.add_argument(
        "-v",
        dest="verbose",
        action="count",
        default=0,
        help="Increase output verbosity (-v for evidence, -vv for full JSON).",
    )
    replay_p.add_argument(
        "--junit-xml",
        metavar="PATH",
        default=None,
        dest="junit_xml",
        help="Write a JUnit-compatible XML report to PATH.",
    )
    replay_p.add_argument(
        "--strict-flaky",
        action="store_true",
        default=False,
        dest="strict_flaky",
        help="Treat FLAKY results as failures (exit code 1, fail in JUnit XML).",
    )
    replay_p.add_argument(
        "--provider",
        metavar="PROVIDER",
        default=None,
        help=(
            "LLM provider to use for execution. "
            "Built-ins: 'anthropic' (default), 'openai'. "
            "Custom: 'my_package.module:MyProvider'."
        ),
    )
    replay_p.add_argument(
        "--judge-provider",
        metavar="PROVIDER",
        default=None,
        dest="judge_provider",
        help=(
            "LLM provider for the LLM judge. "
            "Defaults to --provider (or 'anthropic' if unset). "
            "Built-ins: 'anthropic', 'openai'. "
            "Custom: 'my_package.module:MyProvider'."
        ),
    )
    replay_p.add_argument(
        "--judge-model",
        metavar="MODEL",
        default=None,
        dest="judge_model",
        help=(
            "Model to use for the LLM judge "
            "(default: from sivo.toml or 'claude-haiku-4-5')."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point. Parses *argv* and dispatches to subcommands."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        sys.exit(_cmd_run(args))
    elif args.command == "replay":
        sys.exit(_cmd_replay(args))
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
