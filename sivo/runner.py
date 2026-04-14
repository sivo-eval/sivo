"""Execution engine and eval engine for sivo.

The **execution engine** calls the LLM asynchronously, captures a full
:class:`~sivo.models.ExecutionRecord`, and writes it to the JSONL store.

The **eval engine** runs eval functions against pre-populated
:class:`~sivo.models.EvalCase` objects — it never calls the LLM.

``run_session()`` orchestrates both layers: discovery → (optional LLM call)
→ eval assertions → ``SessionResult``.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from sivo.judge import LLMJudge

from sivo.models import EvalCase, ExecutionInput, ExecutionRecord
from sivo.providers import CompletionResult, Provider
from sivo.providers.anthropic import _calculate_cost  # re-exported for compat
from sivo.store import JsonlStore

__all__ = ["_calculate_cost"]  # keep importable from sivo.runner


# ---------------------------------------------------------------------------
# ExecutionEngine
# ---------------------------------------------------------------------------


class ExecutionEngine:
    """Async LLM execution engine with concurrency control, retry, and timeout.

    Constructs :class:`~sivo.models.ExecutionRecord` objects from each
    :class:`~sivo.models.ExecutionInput`, writes them to the JSONL store
    immediately on completion, and returns them to the caller.

    Args:
        model: Model to use for all calls. Exact string required (D-012).
               Defaults to ``claude-haiku-4-5``.
        concurrency: Max simultaneous LLM calls. Defaults to 10.
        retries: Number of attempts per call (including the first). Defaults
                 to 3. Uses exponential backoff between attempts (1s, 2s, …).
        timeout: Per-call timeout in seconds. Defaults to 30.
        store: :class:`~sivo.store.JsonlStore` instance. A default store
               at ``.sivo`` is created if not provided.
        api_key: API key forwarded to the default :class:`~sivo.providers.anthropic.AnthropicProvider`.
                 Ignored when *provider* is supplied explicitly.
        provider: :class:`~sivo.providers.Provider` instance to use for LLM
                  calls.  When not supplied, an
                  :class:`~sivo.providers.anthropic.AnthropicProvider` is
                  created using *api_key*.
    """

    DEFAULT_MODEL = "claude-haiku-4-5"
    DEFAULT_CONCURRENCY = 10
    DEFAULT_RETRIES = 3
    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        concurrency: int = DEFAULT_CONCURRENCY,
        retries: int = DEFAULT_RETRIES,
        timeout: float = DEFAULT_TIMEOUT,
        store: JsonlStore | None = None,
        api_key: str | None = None,
        provider: Provider | None = None,
    ) -> None:
        self.model = model
        self.concurrency = concurrency
        self.retries = retries
        self.timeout = timeout
        self.store = store or JsonlStore()
        if provider is not None:
            self.provider = provider
        else:
            from sivo.providers.anthropic import AnthropicProvider
            self.provider = AnthropicProvider(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self, spec: ExecutionInput, *, run_id: str
    ) -> ExecutionRecord:
        """Execute a single LLM call for *spec* and return the record.

        The record is written to the JSONL store before being returned.

        Args:
            spec: Input specification for the LLM call.
            run_id: Run identifier used to group records.

        Returns:
            The :class:`~sivo.models.ExecutionRecord` from the completed call.

        Raises:
            TimeoutError: If the call does not complete within ``self.timeout`` seconds.
            Exception: Re-raised from the Anthropic client after exhausting retries.
        """
        return await self._execute_with_retry(spec, run_id)

    async def run_many(
        self,
        specs: Sequence[ExecutionInput],
        *,
        run_id: str | None = None,
    ) -> list[ExecutionRecord]:
        """Execute *specs* concurrently and return all records.

        Calls are batched by the concurrency semaphore. All records are
        written to the same JSONL file keyed by *run_id*.

        Args:
            specs: Input specifications to execute.
            run_id: Shared run identifier. Auto-generated if not provided.

        Returns:
            List of :class:`~sivo.models.ExecutionRecord` in completion order.
        """
        if run_id is None:
            run_id = _generate_run_id()

        semaphore = asyncio.Semaphore(self.concurrency)

        async def _bounded(spec: ExecutionInput) -> ExecutionRecord:
            async with semaphore:
                return await self._execute_with_retry(spec, run_id)

        return list(await asyncio.gather(*(_bounded(s) for s in specs)))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _execute_with_retry(
        self, spec: ExecutionInput, run_id: str
    ) -> ExecutionRecord:
        """Run the LLM call with exponential-backoff retries."""
        last_exc: BaseException = RuntimeError("No attempts made")

        for attempt in range(self.retries):
            try:
                record = await asyncio.wait_for(
                    self._call_provider(spec, run_id),
                    timeout=self.timeout,
                )
                self.store.write(record)
                return record
            except (asyncio.TimeoutError, TimeoutError):
                last_exc = TimeoutError(
                    f"LLM call timed out after {self.timeout}s "
                    f"(attempt {attempt + 1}/{self.retries})"
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc

            if attempt < self.retries - 1:
                backoff = 2**attempt  # 1s, 2s, 4s …
                await asyncio.sleep(backoff)

        raise last_exc

    async def _call_provider(
        self, spec: ExecutionInput, run_id: str
    ) -> ExecutionRecord:
        """Delegate to ``self.provider.complete()`` and build an ``ExecutionRecord``."""
        messages = _build_messages(spec)
        timestamp = datetime.now(timezone.utc).isoformat()

        result: CompletionResult = await self.provider.complete(
            model=self.model,
            system_prompt=spec.system_prompt,
            messages=messages,
            max_tokens=1024,
            timeout=self.timeout,
            extra_params=spec.params or None,
        )

        return ExecutionRecord(
            id=str(uuid.uuid4()),
            timestamp=timestamp,
            run_id=run_id,
            input=spec.input,
            system_prompt=spec.system_prompt,
            conversation=spec.conversation,
            output=result.output,
            model=self.model,
            params=spec.params,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            metadata=spec.metadata,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_run_id() -> str:
    """Generate a unique run identifier with a human-readable timestamp prefix."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{uuid.uuid4().hex[:8]}"


def _build_messages(spec: ExecutionInput) -> list[dict]:
    """Convert an :class:`~sivo.models.ExecutionInput` to Anthropic messages list."""
    if spec.conversation:
        # Multi-turn: use the full conversation history
        return [{"role": msg.role, "content": msg.content} for msg in spec.conversation]

    # Single-turn: wrap input as a user message
    content = spec.input if isinstance(spec.input, str) else str(spec.input)
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Eval engine
# ---------------------------------------------------------------------------


def get_response(case: EvalCase) -> str:
    """Return the LLM output stored in *case*.

    Eval functions should call this helper (or access ``case.output`` directly)
    rather than calling the LLM themselves. The runner injects the correct
    output transparently in both live and replay modes (D-005).

    Args:
        case: The :class:`~sivo.models.EvalCase` passed to an eval function.

    Returns:
        The model response string.
    """
    return case.output


@dataclass
class EvalResult:
    """Outcome of running one eval function against one :class:`~sivo.models.EvalCase`.

    Attributes:
        eval_name:  Name of the eval function (e.g. ``"eval_tone"``).
        record_id:  ID of the :class:`~sivo.models.ExecutionRecord` used.
        passed:     ``True`` if the eval completed without a hard failure.
                    Note: flaky results also set ``passed=True`` (they do not
                    affect the exit code by default).
        flaky:      ``True`` when the eval yielded an indeterminate verdict.
        error:      The exception raised on failure/flaky (``None`` if passed).
    """

    eval_name: str
    record_id: str
    passed: bool
    flaky: bool = False
    error: BaseException | None = None


@dataclass
class SessionResult:
    """Aggregated results for a complete sivo run.

    Attributes:
        run_id:              Run identifier shared by all results.
        results:             All individual eval results, in execution order.
        total_input_tokens:  Sum of input tokens across all execution records.
        total_output_tokens: Sum of output tokens across all execution records.
        total_cost_usd:      Estimated total USD cost for the run.
        cost_by_eval:        Estimated USD cost attributed to each eval function.
    """

    run_id: str
    results: list[EvalResult] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    cost_by_eval: dict[str, float] = field(default_factory=dict)

    @property
    def passed_count(self) -> int:
        """Count of outright-pass results (excludes flaky)."""
        return sum(1 for r in self.results if r.passed and not r.flaky)

    @property
    def flaky_count(self) -> int:
        """Count of flaky results."""
        return sum(1 for r in self.results if r.flaky)

    @property
    def failed_count(self) -> int:
        """Count of outright-fail results."""
        return sum(1 for r in self.results if not r.passed)

    @property
    def all_passed(self) -> bool:
        """``True`` when there are no outright failures (flaky counts as pass)."""
        return self.failed_count == 0

    def is_success(self, *, strict_flaky: bool = False) -> bool:
        """Return ``True`` if the session should be considered a success.

        Args:
            strict_flaky: When ``True``, flaky results are treated as failures
                          and cause this method to return ``False``.
        """
        if self.failed_count > 0:
            return False
        if strict_flaky and self.flaky_count > 0:
            return False
        return True


class EvalEngine:
    """Runs eval functions against pre-populated :class:`~sivo.models.EvalCase` objects.

    The eval engine never calls the LLM. It receives :class:`~sivo.models.EvalCase`
    objects whose ``output`` field has already been populated (either by the
    execution engine in live mode or from a JSONL store in replay mode).
    """

    def run(
        self,
        eval_func: Callable[[EvalCase], None],
        case: EvalCase,
        *,
        eval_name: str,
        record_id: str,
        fixture_kwargs: dict | None = None,
    ) -> EvalResult:
        """Run *eval_func* against *case* and return the result.

        The eval function is expected to raise :class:`~sivo.assertions.EvalAssertionError`
        (or any :class:`AssertionError`) on failure. Any other exception is
        also caught and reported as a failure so the runner can continue.

        Args:
            eval_func:       The eval function to call.
            case:            Pre-populated :class:`~sivo.models.EvalCase`.
            eval_name:       Name for reporting (e.g. ``"eval_tone"``).
            record_id:       ID of the source ``ExecutionRecord``.
            fixture_kwargs:  Optional resolved fixture values to pass as
                             keyword arguments after ``case``.

        Returns:
            :class:`EvalResult` with ``passed=True`` or failure details.
        """
        from sivo.assertions import FlakyEvalError

        kwargs = fixture_kwargs or {}
        try:
            eval_func(case, **kwargs)
            return EvalResult(eval_name=eval_name, record_id=record_id, passed=True)
        except FlakyEvalError as exc:
            return EvalResult(
                eval_name=eval_name,
                record_id=record_id,
                passed=True,
                flaky=True,
                error=exc,
            )
        except BaseException as exc:  # noqa: BLE001
            return EvalResult(
                eval_name=eval_name,
                record_id=record_id,
                passed=False,
                error=exc,
            )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_session(
    path: Path,
    *,
    run_id: str | None = None,
    store: JsonlStore | None = None,
    eval_filter: str | None = None,
    metadata_filter: dict[str, str] | None = None,
    fail_fast: bool = True,
    on_result: Callable[[EvalResult], None] | None = None,
    pdb_hook: Callable | None = None,
    judge: "LLMJudge | None" = None,
) -> SessionResult:
    """Discover eval functions and run them against stored records or case generators.

    For regular evals (no ``eval_*_cases`` companion), loads
    :class:`~sivo.models.ExecutionRecord` objects from the JSONL store for
    *run_id* and runs every eval function against every record.

    For data-driven evals (with an ``eval_*_cases`` companion function), calls
    the companion to obtain :class:`~sivo.models.EvalCase` objects directly
    — no store access is required for those evals.

    Fixtures declared with ``@sivo.fixture`` are injected automatically
    based on eval function parameter names.

    Args:
        path:            File or directory to discover eval functions in.
        run_id:          Run identifier to load records from. Auto-generated
                         when not provided (required for non-data-driven evals).
        store:           :class:`~sivo.store.JsonlStore` instance. Defaults
                         to ``./.sivo``.
        eval_filter:     If given, only run the eval function with this exact
                         name.
        metadata_filter: If given, only run against records whose metadata
                         matches all key-value pairs (record-based evals only).
        fail_fast:       Stop on the first failure (default). Set to ``False``
                         to run all evals regardless of failures.
        on_result:       Optional callback called immediately after each
                         :class:`EvalResult` is produced. Used by the CLI for
                         live progress display.
        pdb_hook:        Optional ``--pdb-llm`` hook. Called on every failing
                         (non-flaky) result with ``(result, case, eval_func)``
                         and must return ``(action, result)`` where *action* is
                         ``"skip"``, ``"continue"``, or ``"abort"``. When set,
                         *fail_fast* is suppressed while the hook is active.
        judge:           Optional :class:`~sivo.judge.LLMJudge` instance to
                         use as the session-level judge override.  When provided,
                         all ``assert_judge()`` calls within eval functions use
                         this judge instead of the default. Supports
                         ``--judge-provider`` / ``--judge-model`` CLI flags.

    Returns:
        :class:`SessionResult` containing all :class:`EvalResult` objects.

    Raises:
        ValueError: If no records are found for *run_id* (record-based evals).
        ValueError: If no eval functions are discovered in *path*.
        ValueError: If *run_id* is required (non-data-driven evals) but absent.
    """
    from sivo.discovery import discover
    from sivo.fixtures import collect_fixtures
    from sivo.judge import set_session_judge

    # Install the session-level judge override so that assert_judge() calls
    # inside eval functions use the requested provider/model (if any).
    _prev_judge = set_session_judge(judge)
    try:
        # Generate a run_id if not provided (used for data-driven-only runs)
        _auto_run_id = run_id is None
        if run_id is None:
            run_id = _generate_run_id()

        _store = store or JsonlStore()
        engine = EvalEngine()

        # 1. Discover eval functions
        evals = discover(path, eval_filter=eval_filter)
        if not evals:
            raise ValueError(
                f"No eval functions found in {path!r}"
                + (f" matching --eval {eval_filter!r}" if eval_filter else "")
            )

        # 2. Load records only for evals that need them (no cases_func)
        record_evals = [e for e in evals if e.cases_func is None]
        records: list = []
        if record_evals:
            if _auto_run_id:
                raise ValueError(
                    "Some eval functions have no companion cases generator — "
                    "a --run-id is required to load stored records."
                )
            if metadata_filter:
                records = _store.filter(run_id, **metadata_filter)
            else:
                records = _store.read(run_id)
            if not records:
                raise ValueError(f"No records found for run_id={run_id!r}")

        # 3. Aggregate token/cost stats from the records (data-driven evals = $0)
        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        total_cost = sum(r.cost_usd for r in records)

        # Per-eval cost: split record costs only across the record-based evals.
        n_record_evals = len(record_evals)
        cost_by_eval: dict[str, float] = {e.name: 0.0 for e in evals}
        if n_record_evals > 0:
            for record in records:
                per_eval = record.cost_usd / n_record_evals
                for discovered in record_evals:
                    cost_by_eval[discovered.name] += per_eval

        session = SessionResult(
            run_id=run_id,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cost_usd=total_cost,
            cost_by_eval=cost_by_eval,
        )

        # 4. Collect fixtures from eval modules
        registry = collect_fixtures(evals)

        # 5. Run (eval_function × cases) pairs — outer loop is over evals so that
        #    eval-scoped fixtures are initialised once per eval function.
        _active_pdb_hook = pdb_hook
        registry.initialize_session()
        try:
            for discovered in evals:
                # Build the (record_id, case) sequence for this eval
                if discovered.cases_func is not None:
                    # Data-driven: cases come from the companion generator
                    raw_cases = list(discovered.cases_func())
                    items: list[tuple[str, EvalCase]] = [
                        (f"case-{i}", c) for i, c in enumerate(raw_cases)
                    ]
                else:
                    # Record-based: convert ExecutionRecords to EvalCases
                    items = [(r.id, r.to_eval_case()) for r in records]

                registry.initialize_eval()
                try:
                    for record_id, case in items:
                        fixture_kwargs = registry.resolve(discovered.func)
                        result = engine.run(
                            discovered.func,
                            case,
                            eval_name=discovered.name,
                            record_id=record_id,
                            fixture_kwargs=fixture_kwargs,
                        )

                        # pdb-llm: pause on outright failures when hook is active
                        _pdb_handled = False
                        if (
                            _active_pdb_hook is not None
                            and not result.passed
                            and not result.flaky
                        ):
                            action, result = _active_pdb_hook(result, case, discovered.func)
                            _pdb_handled = True
                            if action == "abort":
                                session.results.append(result)
                                if on_result is not None:
                                    on_result(result)
                                return session
                            elif action == "continue":
                                _active_pdb_hook = None

                        session.results.append(result)
                        if on_result is not None:
                            on_result(result)

                        if (
                            not _pdb_handled
                            and _active_pdb_hook is None
                            and fail_fast
                            and not result.passed
                        ):
                            return session
                finally:
                    registry.teardown_eval()
        finally:
            registry.teardown_session()

        return session
    finally:
        set_session_judge(_prev_judge)
