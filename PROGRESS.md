# PROGRESS.md — Sivo

Update this file at the end of every Claude Code session. It is the primary continuity mechanism across sessions.

---

## Current status

**Phase:** Rename evalkit → sivo — Complete
**State:** Complete. 520 tests pass, 7 skipped.

---

## Phase overview

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Project scaffold + data models | Complete |
| 2 | Assertion library + LLM judge | Complete |
| 3 | Execution engine + JSONL store | Complete |
| 4 | Eval discovery + runner orchestration | Complete |
| 5 | Terminal UI + session receipt | Complete |
| 6 | Replay system | Complete |
| 7 | Interactive REPL (`--pdb-llm`) | Complete |
| 8 | CI integration + JUnit XML | Complete |
| 9 | Fixtures + data-driven evals | Complete |
| 10 | Examples, docs, PyPI release | Complete |

---

## E2e testing approach

Each phase from 3 onward includes 1–2 e2e scenarios that exercise new functionality in a realistic usage pattern. Conventions:

- **Location:** `tests/e2e/` (excluded from nothing — runs as part of `uv run pytest` unless guarded by `--no-llm`)
- **No API calls:** All e2e tests use canned JSONL `ExecutionRecord` fixtures via the `get_response()` replay pattern
- **Self-contained:** Each scenario ships its own eval file, fixture data, and expected outcomes — no shared mutable state
- **Full pipeline:** CLI invocation → eval execution → output verification (exit codes, terminal output, JUnit XML where relevant)
- **Performance tests:** Live in `tests/perf/` and are excluded from the default `pytest` run via a custom marker (`perf`)

---

## Phase 1 — Project scaffold + data models

**Goal:** Installable package with all core data models defined, validated, and tested. No LLM calls yet.

### Tasks

- [x] Initialise repo with `uv init`
- [x] Set up `pyproject.toml` with dependencies (pydantic, rich, anthropic, pytest)
- [x] Create package structure (`sivo/` with all module stubs)
- [x] Implement `EvalCase` model with full field validation
- [x] Implement `ExecutionRecord` model with full field validation
- [x] Implement `JudgeVerdict` model
- [x] Implement `Trace` and `Step` models (reserved, read-only)
- [x] Implement `Message` model for conversation history
- [x] Write unit tests for all models (field validation, serialisation, edge cases)
- [x] Confirm `uv run pytest` passes cleanly

### Completed
All Phase 1 tasks complete. 48 tests passing.

### Notes
- Used `dependency-groups.dev` (not deprecated `tool.uv.dev-dependencies`) for pytest/pytest-asyncio.
- All models in `sivo/models.py`. All module stubs created.
- `ExecutionRecord.to_eval_case()` merges record metadata with `run_id` into the resulting `EvalCase`.
- `Message.role` constrained to `Literal["user", "assistant", "system"]`.
- `input_tokens`, `output_tokens` use `ge=0`; `cost_usd` uses `ge=0.0` for non-negative enforcement.

---

## Phase 2 — Assertion library + LLM judge

**Goal:** All assertion functions working, LLM judge returning structured verdicts, caching in place.

### Tasks

- [x] Implement deterministic assertions: `assert_contains`, `assert_not_contains`, `assert_matches_schema`, `assert_regex`, `assert_length`
- [x] Implement `AssertionError` subclass that carries structured context (which assertion, what failed, evidence)
- [x] Implement `LLMJudge` class with `assess()` method
- [x] Implement judge prompt template (rubric + state summary → structured `JudgeVerdict`)
- [x] Implement built-in rubrics: `helpfulness`, `tone`, `toxicity`, `factual_consistency`, `conciseness`
- [x] Implement content-hash caching for judge calls (SHA256 of rubric + output)
- [x] Implement `assert_judge()` public API
- [x] Write unit tests for all deterministic assertions
- [x] Write integration tests for `LLMJudge` (can be skipped in CI with `--no-llm`)

### Completed
All Phase 2 tasks complete. 107 tests passing, 4 skipped (live LLM integration tests — run without `--no-llm` to execute).

### Notes
- `EvalAssertionError` subclasses `AssertionError`; carries `assertion_type`, `expected`, `actual`.
- `assert_matches_schema` accepts Pydantic `BaseModel` subclasses or any type supported by `pydantic.TypeAdapter` (e.g. `list[str]`). No new dependencies added.
- `LLMJudge` uses tool_use (`record_verdict` tool) for structured output — reliable across all Claude models.
- Cache is session-scoped (in-memory dict on the `LLMJudge` instance). Built-in rubric name is resolved to full text before hashing (so name and full text share the same cache entry).
- `assert_judge()` uses a module-level default judge instance shared across calls in a session.
- Added `tests/conftest.py` with `--no-llm` flag. Added `[tool.pytest.ini_options] markers` to `pyproject.toml` for the `integration` mark.

---

## Phase 3 — Execution engine + JSONL store

**Goal:** Can call an LLM, capture a full `ExecutionRecord`, and write it to JSONL.

### Tasks

- [x] Implement `store.py`: JSONL write, read, list runs, filter by run_id
- [x] Implement `ExecutionEngine`: async LLM call, token/cost capture, record construction
- [x] Implement concurrency control (`--concurrency N`, default 10)
- [x] Implement retry with exponential backoff (default 3 attempts)
- [x] Implement per-call timeout (default 30s)
- [x] Write unit tests with mocked LLM client
- [x] Write integration test: real LLM call → JSONL record round-trip (guarded by `--no-llm`)

### Completed
All Phase 3 tasks complete. 144 tests passing, 5 skipped (live LLM integration tests).

### Notes
- Added `ExecutionInput` to `models.py` — the input spec for the execution engine (no `output` field, unlike `EvalCase`). This cleanly separates "what the engine takes" from "what eval functions receive".
- `ExecutionEngine` uses `anthropic.AsyncAnthropic` with `asyncio.Semaphore` for concurrency control.
- Retry uses `asyncio.sleep(2**attempt)` backoff; timeout via `asyncio.wait_for`.
- Cost table in `runner.py` covers all current Claude model strings. Unknown models fall back to Sonnet pricing.
- `store.py` is path-agnostic via `JsonlStore(store_path)` — tests pass `tmp_path` to avoid polluting the working directory.

---

## Phase 4 — Eval discovery + runner orchestration

**Goal:** `sivo run` discovers and executes eval functions end-to-end.

### Tasks

- [x] Implement `discovery.py`: find `eval_*` functions in `eval_*.py` files
- [x] Implement `runner.py`: orchestrate execution engine + eval engine
- [x] Implement `get_response()` injection pattern (live vs replay)
- [x] Implement fail-fast logic (`--fail-fast` / `--no-fail-fast`)
- [x] Implement basic CLI: `sivo run [path]`
- [x] Write unit tests: discovery, runner orchestration, `get_response()` injection
- [x] **E2e — CLI against canned JSONL:** `tests/e2e/test_cli_jsonl.py`
  - Canned fixture: a `run_<id>.jsonl` file with 3 `ExecutionRecord` entries (2 pass, 1 fail)
  - Scenario A: all-pass JSONL → `sivo run` exits 0, all evals reported as passed
  - Scenario B: mixed JSONL → exits 1, failed eval identified by name in output
  - Verifies the CLI can be invoked as a subprocess and the exit code is correct
- [x] **E2e — Discovery & filtering:** `tests/e2e/test_discovery_filtering.py`
  - Mini project: `evals/eval_tone.py`, `evals/eval_accuracy.py`, `evals/sub/eval_edge_cases.py`; each file contains one `eval_*` function using canned JSONL fixtures
  - Scenario A: `sivo run evals/` discovers and runs all 3 eval functions; verifies count of 3 in output
  - Scenario B: `sivo run evals/eval_tone.py` runs only that file's eval; verifies count of 1 in output
  - Scenario C: `sivo run --eval eval_edge_cases` runs only the matching eval regardless of directory depth; verifies count of 1 in output

### Completed
All Phase 4 tasks complete. 212 tests passing, 5 skipped (live LLM integration tests).

### Notes
- `discovery.py` uses `importlib.util.spec_from_file_location` with a deterministic module name to load eval files. Module names are cached in `sys.modules` to avoid double-loading the same file.
- `eval_X_cases` is a cases-generator companion only when a sibling `eval_X` function exists in the same module. Functions whose names end in `_cases` but have no sibling (e.g. `eval_edge_cases`) are treated as standalone evals. This required a two-pass approach in `load_eval_functions`.
- `runner.py` now contains both `ExecutionEngine` (Phase 3, calls LLM) and `EvalEngine` + `run_session` (Phase 4, replay-mode orchestration). Live mode (without `--run-id`) returns exit 2 with an explanatory message — requires case generators (Phase 9).
- `get_response(case)` exported from `sivo/__init__.py`; returns `case.output`.
- CLI uses `argparse` (no new dependency needed). Default is fail-fast; `--no-fail-fast` runs all evals.
- `sivo run` path argument resolves relative to CWD, so subprocess e2e tests set `cwd=tmp_path` to keep fixtures self-contained.

---

## Phase 5 — Terminal UI + session receipt

**Goal:** Rich terminal output with per-eval results and a session-level summary receipt.

### Tasks

- [x] Implement verbosity levels: default (pass/fail + one-sentence reason), `-v` (failure context + evidence), `-vv` (full `JudgeVerdict` JSON + prompts)
- [x] Implement `report.py`: session receipt (counts, token totals, cost, cost-per-eval, run id)
- [x] Implement per-eval cost attribution
- [x] Implement progress display during execution (Rich live display or progress bar)
- [x] Write unit tests for receipt formatting and cost aggregation
- [x] **E2e — Output format correctness:** `tests/e2e/test_output_formats.py`
  - Canned fixture: JSONL with mixed results (pass, fail, flaky-flagged)
  - Verifies default output contains pass/fail counts and run id
  - Verifies `-v` output includes failure evidence text
  - Verifies `-vv` output includes full `JudgeVerdict` JSON fields

### Completed
All Phase 5 tasks complete. 261 tests passing, 5 skipped (live LLM integration tests).

### Notes
- `FlakyEvalError` added to `assertions.py`. `EvalEngine.run()` catches it before the generic `BaseException` handler and sets `passed=True, flaky=True` on the result (D-008: flaky doesn't fail CI by default).
- `EvalResult` gains `flaky: bool = False`. `SessionResult` gains `passed_count` (excludes flaky), `flaky_count`, `failed_count`, `all_passed` (True when no outright fails), plus token/cost stats (`total_input_tokens`, `total_output_tokens`, `total_cost_usd`, `cost_by_eval`).
- Per-eval cost: each record's cost is split equally across the N eval functions that ran on it, so `sum(cost_by_eval.values()) == total_cost_usd`.
- `run_session` accepts `on_result` callback; the CLI uses this to stream each result to the console immediately as it completes (progress display without a separate progress-bar widget).
- `report.py` uses Rich (`Console`, `Rule`). `print_result` handles v0/v1/v2 by inspecting `error.assertion_type` and `error.actual` (a `JudgeVerdict`) on `EvalAssertionError`. `make_console()` disables Rich's syntax highlighting to avoid misinterpreting eval output text as markup.
- `cli.py` now delegates all formatting to `report.py`. `_print_session` removed.

---

## Phase 6 — Replay system

**Goal:** `sivo replay <run_id>` re-runs eval assertions against stored records at zero API cost.

### Tasks

- [x] Implement `replay.py`: load `ExecutionRecord` objects from JSONL, convert to `EvalCase`, run eval engine
- [x] Implement `sivo replay <run_id>` CLI command
- [x] Implement `--eval <name>` filter (replay specific eval only)
- [x] Implement `--filter key=value` metadata filter
- [x] Write unit tests for replay runner (record loading, filter logic)
- [x] **E2e — Live vs replay parity:** `tests/e2e/test_replay_parity.py`
  - Canned fixture: a JSONL with 4 records (mix of pass and fail outcomes)
  - Run `sivo replay` against the fixture; assert results match expected outcomes
  - Run again with `--eval <name>` filter; assert only that eval's result is reported
  - Verifies zero LLM calls are made (mock/assert `anthropic.Anthropic` is never instantiated)

### Completed
All Phase 6 tasks complete. 293 tests passing, 5 skipped (live LLM integration tests).

### Notes
- `replay.py` contains `replay_session()` (thin wrapper around `run_session`) and `parse_filters()` (parses `KEY=VALUE` strings to dict; `partition("=")` handles values containing `=`).
- `run_session` in `runner.py` gains `metadata_filter: dict[str, str] | None = None`. When provided, calls `store.filter(run_id, **metadata_filter)` instead of `store.read(run_id)`.
- `sivo replay <run_id> [path]` CLI subcommand: `run_id` is a positional arg (required), `path` is optional (defaults to `.`). Supports `--eval NAME`, `--filter KEY=VALUE` (repeatable), `--no-fail-fast`, `--store-path`, `-v`.
- No LLM calls during replay — verified in unit tests via `patch("anthropic.Anthropic")` + `patch("anthropic.AsyncAnthropic")` and in e2e by running without `ANTHROPIC_API_KEY` in the environment.

---

## Phase 7 — Interactive REPL (`--pdb-llm`)

**Goal:** Drop into an interactive session on a failing eval to inspect and hot-swap the prompt.

### Tasks

- [x] Implement `repl.py`: pause on failing eval, expose `input`, `system_prompt`, `output`, `judge_verdict`
- [x] Implement REPL commands: `inspect`, `retry`, `skip`, `continue`, `abort`
- [x] Implement hot-swap: edit `system_prompt` in-REPL and re-run current eval
- [x] Integrate `--pdb-llm` flag into the CLI runner
- [x] Write unit tests for REPL command dispatch and hot-swap logic (headless / non-interactive)

### Completed
All Phase 7 tasks complete. 331 tests passing, 5 skipped (live LLM integration tests).

### Notes
- `repl.py` exports `PdbLlmSession` (the REPL loop) and `make_pdb_hook` (factory for use with `run_session`).
- All I/O is injected: `console: Console` and `input_fn: Callable[[], str]`. Tests use `Console(file=StringIO())` and an iterator-backed `input_fn`.
- Commands: `inspect` (prints case fields + judge verdict), `retry` (re-runs eval, updates `self._result`), `skip` (returns `"skip"`, REPL stays active), `continue` (returns `"continue"`, REPL disabled), `abort` (returns `"abort"`, run stops). `EOFError`/`KeyboardInterrupt` → `"abort"`.
- Hot-swap: `system_prompt = "value"` (double or single quotes, or bare) calls `case.model_copy(update=...)`. Settable fields: `system_prompt` only (v1).
- `run_session` gains `pdb_hook: Callable | None = None`. The hook is invoked on non-flaky failures; returns `(action, final_result)`. Strings used instead of an enum to avoid circular imports. `_pdb_handled` flag prevents `fail_fast` from triggering on results that the user already acknowledged via the hook.
- `run` subcommand gains `--pdb-llm` flag; creates hook via `make_pdb_hook(console=console)`.
- Bug found during testing: after hook returns `"continue"`, `fail_fast` must NOT apply to that same result (user already acknowledged it). Fixed with `_pdb_handled` boolean.

---

## Phase 8 — CI integration + JUnit XML

**Goal:** Machine-readable output and correct exit codes for CI pipelines.

### Tasks

- [x] Implement `--junit-xml <path>`: write JUnit XML on every run
- [x] Implement JSON summary: write `.sivo/results/<run_id>.json` on every run
- [x] Implement exit code contract: 0 = all pass, 1 = any fail, 2 = internal error
- [x] Implement `--strict-flaky`: treat `FLAKY` as `FAIL`
- [x] Write unit tests for XML serialisation and exit code logic
- [x] **E2e — JUnit XML well-formedness:** `tests/e2e/test_junit_xml.py`
  - Canned fixture: JSONL with 2 passing and 1 failing eval
  - Run `sivo run --junit-xml <tmp_path>/results.xml` as subprocess
  - Parse the output XML with `xml.etree.ElementTree`; assert it is valid and contains the correct `<testcase>` entries
  - Assert failing eval appears as `<failure>` element; passing evals have no `<failure>` child
  - Assert exit code is 1 (due to failing eval)

### Completed
All Phase 8 tasks complete. 389 tests passing, 5 skipped (live LLM integration tests).

### Notes
- `SessionResult.is_success(strict_flaky=False)` added alongside `all_passed`. `all_passed` preserved for existing callers; CLI now uses `is_success()` for exit code logic.
- `write_junit_xml(session, path, *, strict_flaky=False)` in `report.py`: stdlib `xml.etree.ElementTree` only (no new deps). Structure: `<testsuites><testsuite>` with `<testcase>` per result. PASS → no children; FAIL → `<failure type=... message=...>`; FLAKY → `<skipped message="FLAKY: ...">` by default, `<failure>` with `strict_flaky=True`.
- `write_json_summary(session, store_path)` writes `{store_path}/results/{run_id}.json` automatically after every `run` and `replay`. Fields: run_id, passed/failed/flaky/total counts, token totals, cost, per-result array.
- Both `run` and `replay` subcommands gain `--junit-xml PATH` and `--strict-flaky`. JSON summary is always written; `--junit-xml` is opt-in.
- `testcase` name format: `{eval_name}[{record_id}]` — uniquely identifies each eval×record pair. `classname` = `run_id` (groups all tests from a run).

---

## Phase 9 — Fixtures + data-driven evals

**Goal:** `@sivo.fixture` decorator and data-driven eval functions with case generators.

### Tasks

- [x] Implement `fixtures.py`: `@sivo.fixture(scope="session"|"eval")` decorator
- [x] Implement fixture injection into eval function signatures (inspect parameter names, resolve from fixture registry)
- [x] Implement session-scoped fixture lifecycle (initialise once per run, teardown after)
- [x] Implement eval-scoped fixture lifecycle (initialise once per eval function, teardown after)
- [x] Implement data-driven eval pairing: `eval_*_cases()` companion generator → `eval_*()` function
- [x] Write unit tests for fixture scoping, lifecycle, and injection
- [x] Write unit tests for case generator pairing and data-driven expansion
- [x] **E2e — Fixture scoping:** `tests/e2e/test_fixture_scoping.py`
  - Mini project with a session-scoped fixture (a counter or shared object) and two eval functions that both use it
  - Canned JSONL records (no API calls)
  - Verifies the session fixture is initialised exactly once across both evals
  - Verifies an eval-scoped fixture is initialised once per eval function
- [x] **E2e — Data-driven evals:** `tests/e2e/test_data_driven.py`
  - Mini project with `eval_summariser_cases()` returning 3 `EvalCase` objects
  - `eval_summariser()` runs assertions on each case
  - Verifies all 3 cases are discovered and run individually
  - Verifies a failure in case 2 does not prevent case 3 from running (with `--no-fail-fast`)

### Completed
All Phase 9 tasks complete. 434 tests passing, 5 skipped (live LLM integration tests). 3 perf tests available via `uv run pytest tests/perf/ -m perf`.

### Notes
- `fixtures.py` exports `fixture(*, scope="session"|"eval")` decorator (stamps `__sivo_fixture_scope__`), `FixtureRegistry`, and `collect_fixtures(evals)`.
- `FixtureRegistry` manages two caches (`_session_cache`, `_eval_cache`) and two generator lists for teardown. `resolve(func)` inspects the function signature, skips the first parameter (`case`), and looks up remaining params from the caches.
- Generator fixtures (yield-based): `_call_factory` detects generators via `inspect.isgeneratorfunction`; `next(gen)` advances to the yield point for setup, advancing once more during teardown catches `StopIteration`.
- `collect_fixtures(evals)` scans `get_loaded_module()` for each eval's source file and picks up any object with a `__sivo_fixture_scope__` attribute.
- `discovery.py` gains `get_loaded_module(source_file)` (public) and `_module_name(source_file)` (extracted helper) so fixtures can look up already-loaded modules without re-importing.
- `run_session` loop reordered from `(record × eval)` to `(eval × record)`: required for eval-scoped fixture correctness. Session lifecycle wraps the outer loop; eval lifecycle wraps each eval's record iteration.
- `run_id` made optional in `run_session`. When `None` and regular (non-data-driven) evals exist, raises `ValueError("...run-id is required...")`. When `None` and only data-driven evals exist, auto-generates a UUID run_id.
- Data-driven case IDs are `case-0`, `case-1`, … (index-based). Mixed sessions (data-driven + record-based) work in the same file.

---

## Phase 10 — Examples, docs, PyPI release

**Goal:** Polished, publishable package with working examples and documentation.

### Tasks

- [x] Implement `sivo.toml` configuration loading (default model, concurrency, timeout, store path, judge settings, cost warn threshold)
- [x] Write `examples/eval_refund_policy.py` — realistic eval against a canned JSONL fixture
- [x] Write `examples/eval_tone.py` — eval using `assert_judge` with built-in rubric
- [x] Write README (install, quickstart, concepts, CLI reference)
- [x] Audit and clean up all public API surfaces
- [x] Verify `uv run pytest` is fully green
- [x] **E2e — Packaging smoke test:** `tests/e2e/test_packaging.py`
  - Creates a temporary directory with a minimal `eval_smoke.py` file and a canned JSONL fixture
  - Installs sivo into a fresh `venv` via `pip install <wheel>` (built with `uv build`)
  - Runs `sivo run eval_smoke.py` inside that venv as a subprocess
  - Asserts exit code 0 and that the session receipt appears in stdout

### Completed
All Phase 10 tasks complete. 449 tests passing, 5 skipped (live LLM integration tests). 2 packaging tests available via `uv run pytest tests/e2e/test_packaging.py -m packaging`.

### Notes
- `sivo/config.py` — `load_config(search_path)` walks upward from search_path looking for `sivo.toml`; returns all-default `SivoConfig` when not found. Fields: `default_model`, `concurrency`, `timeout`, `store_path`, `judge_model`, `judge_retry_attempts`, `cost_warn_above_usd`. Uses stdlib `tomllib` (Python 3.11+, no new dep).
- CLI (`run` and `replay`) now loads config and: (a) uses `config.store_path` when `--store-path` is not provided, (b) emits a cost warning via `report.print_cost_warning` when session cost exceeds `config.cost_warn_above_usd`.
- `sivo/__init__.py` now exports all public symbols: `fixture`, `get_response`, `EvalCase`, `ExecutionRecord`, `JudgeVerdict`, `EvalResult`, `SessionResult`, all assertions, `EvalAssertionError`, `FlakyEvalError`, `SivoConfig`, `load_config`.
- `examples/eval_refund_policy.py` — three deterministic assertions against a customer support refund policy use-case. `examples/make_fixture.py` generates canned JSONL for replay.
- `examples/eval_tone.py` — data-driven eval with `assert_judge` (tone rubric), custom rubric (empathy), and session-scoped fixture; can run without stored records.
- `README.md` written: install, quickstart, architecture, assertions, fixtures, data-driven evals, CLI reference table, CI integration, config, REPL.
- D-014 provider isolation audit complete: all Anthropic-specific code confirmed to live only in `ExecutionEngine._call_llm()` and `LLMJudge._call_judge()` / `_extract_tool_input()`. No provider coupling leaked into Phases 4–9. DECISIONS.md updated with current line numbers and refactor scope.
- Packaging test: uses stdlib `venv`, `pip install <wheel>`, subprocess CLI call. Marked `packaging` and excluded from default run via `addopts = "--ignore=tests/e2e/test_packaging.py"`. `pyproject.toml` `markers` table updated.

---

## Multi-LLM Support

See `docs/MULTI_LLM_SPEC.md` for the full spec.

### Phase overview

| Phase | Focus | Status |
|-------|-------|--------|
| A | Provider protocol + registry + AnthropicProvider refactor | Complete |
| B | OpenAI provider | Complete |
| C | CLI + config wiring + e2e | Complete |

---

### Phase C — CLI + config wiring + e2e

**Goal:** Wire `--provider` / `--judge-provider` CLI flags, honour `sivo.toml` provider config, update cache key to include model, write e2e tests, update README, resolve D-014. All 520 tests pass.

#### Tasks

- [x] Add `--provider`, `--judge-provider`, `--judge-model` to both `run` and `replay` subparsers
- [x] Implement `_resolve_judge()` helper in `cli.py` — resolves CLI flags + config → `LLMJudge | None`
- [x] Add `judge: LLMJudge | None` param to `run_session` and `replay_session`; save/restore module-level `_session_judge_override` via `set_session_judge()`
- [x] Add `set_session_judge()` to `judge.py`; update `_get_default_judge` to return override when set
- [x] Update `LLMJudge._cache_key` to include model (was `SHA256(rubric + output)`, now `SHA256(model + rubric + output)`)
- [x] Update `LLMJudge.assess` to pass `self.model` to `_cache_key`
- [x] Update `sivo/providers/__init__.py` docstring to mention OpenAI + custom providers
- [x] Write `tests/e2e/test_multi_provider.py` — 11 tests covering: anthropic + openai judge provider mocks, same-pipeline-structure parity, custom provider via import path, CLI `--judge-provider` + `--judge-model` flags, `sivo.toml` provider config, unknown provider exits 2
- [x] Update `tests/test_judge.py` — 8 new tests for: model-aware cache key, `set_session_judge` / session override lifecycle
- [x] Update `README.md` — "Multi-provider support" section with provider table, TOML config examples, custom provider walkthrough, updated options table
- [x] Update `DECISIONS.md` — D-014 resolved with Phase A–C summary
- [x] Confirm 520 tests pass (19 new tests added)

#### Notes

- `_resolve_judge()` in `cli.py` returns `None` (no-op) when both judge provider and model match the built-in defaults, avoiding an unnecessary import.
- The precedence for judge provider: CLI `--judge-provider` > CLI `--provider` > `[sivo.judge] provider` in TOML > `[sivo] provider` in TOML > `"anthropic"`.
- `set_session_judge(judge)` returns the previous value so `run_session` can restore it in a `finally` block — correct even if an exception is raised mid-session.
- Cache key change is backwards-compatible: `_cache_key(rubric, output)` (2-arg form) still works due to `model=""` default.
- Session judge override (`_session_judge_override`) takes precedence over the model-matching logic in `_get_default_judge`, so a custom judge is always used regardless of the `model` argument passed to `assert_judge`.

---

### Phase B — OpenAI provider

**Goal:** Implement `OpenAIProvider`, prompt-based JSON fallback judge, and add `openai` as an optional dependency. All 501 tests pass.

#### Tasks

- [x] Add `openai>=1.0` to `[project.optional-dependencies]` (`pip install sivo[openai]`) and `[dependency-groups] dev`
- [x] Create `sivo/providers/_fallback_judge.py` — `build_fallback_system_prompt()` + `parse_fallback_response()` for providers without native function calling
- [x] Create `sivo/providers/openai.py` — `OpenAIProvider` with `complete()` (async, `chat.completions.create`) and `judge()` (sync, function-calling via `tools`/`tool_choice`)
- [x] Update `sivo/providers/registry.py` — add lazy `"openai"` resolution with clear `ImportError` if `openai` not installed; `_BUILTIN_PROVIDER_NAMES` frozenset for error messages
- [x] Fix `tests/test_providers.py` — `test_get_provider_unknown_builtin_raises` updated to use `"no-such-provider"` instead of `"openai"`
- [x] Create `tests/test_providers_openai.py` — unit tests with mocked `openai.AsyncOpenAI` / `openai.OpenAI`
- [x] Create `tests/test_fallback_judge.py` — tests for fallback judge utility
- [x] Integration test class `TestOpenAIProviderIntegration` in `test_providers_openai.py` (guarded by `--no-llm` + `OPENAI_API_KEY` check)
- [x] Confirm 501 tests pass (28 new tests added)

#### Notes

- `sivo/providers/openai.py` — `OpenAIProvider` mirrors `AnthropicProvider` shape. System prompt prepended as `{"role": "system", ...}` message (OpenAI has no top-level `system` param). Tokens use `prompt_tokens`/`completion_tokens`. Judge uses `tools=[{"type": "function", ...}]` + `tool_choice={"type": "function", "function": {"name": "record_verdict"}}`. Cost table covers GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo, o1, o3 families.
- `sivo/providers/_fallback_judge.py` — `build_fallback_system_prompt(base)` appends JSON schema to base prompt. `parse_fallback_response(raw_text)` strips markdown fences, parses JSON, validates required fields, returns `JudgeVerdict`. Used for providers that lack native tool/function calling.
- Registry lazily loads `OpenAIProvider` via `_get_openai_provider_class()` to avoid import-time failures when `openai` is not installed.
- `openai>=1.0` added as optional extra (`pip install sivo[openai]`) and dev dependency.

---

### Phase A — Provider protocol + registry

**Goal:** Extract all Anthropic-specific code into `AnthropicProvider`. Zero behaviour change — all existing tests pass.

#### Tasks

- [x] Create `sivo/providers/__init__.py` with `Provider` protocol and `CompletionResult`
- [x] Create `sivo/providers/registry.py` with `get_provider()` resolution
- [x] Create `sivo/providers/anthropic.py` — extract existing code from execution engine and judge into `AnthropicProvider`
- [x] Update `SivoConfig` with `provider` and `judge_provider` fields
- [x] Update execution engine to use `provider.complete()` instead of `_call_llm()`
- [x] Update judge to use `provider.judge()` instead of `_call_judge()`
- [x] Write unit tests for providers package (`tests/test_providers.py`)
- [x] Update `test_judge.py` cache tests to patch `judge._provider.judge` instead of removed `_call_judge`
- [x] Confirm all 473 tests pass (zero behaviour change)

#### Notes

- `sivo/providers/__init__.py` — `CompletionResult` (frozen dataclass: `output`, `input_tokens`, `output_tokens`, `cost_usd`, `model`, `raw_response`). `Provider` is a `@runtime_checkable Protocol` with `name: str`, `async complete(...)`, and `judge(...)` (sync — matches existing synchronous judge behaviour).
- `sivo/providers/anthropic.py` — `AnthropicProvider` wraps `anthropic.AsyncAnthropic` (in `complete`) and `anthropic.Anthropic` (in `judge`). Contains cost table, `_calculate_cost`, and `_build_judge_tool` (all moved from `runner.py` / `judge.py`). `_calculate_cost` remains importable from `sivo.runner` via re-export for backwards compatibility.
- `sivo/providers/registry.py` — `get_provider(name, *, api_key)` resolves built-ins by name (`"anthropic"`) and custom providers by dotted import path (`"my_pkg.mod:ClassName"`). Raises `ValueError` for unknown built-ins, `TypeError` if the resolved class fails the `Provider` protocol check.
- `runner.py` changes: removed `_COST_TABLE`, `_FALLBACK_COST`, `_calculate_cost`, `_call_llm`. Added `provider: Provider | None` param to `ExecutionEngine.__init__`; defaults to `AnthropicProvider(api_key=...)`. New `_call_provider()` method calls `self.provider.complete()` and builds the `ExecutionRecord`. `_calculate_cost` re-exported for compat.
- `judge.py` changes: removed `_build_judge_tool`, `_call_judge`, `_extract_tool_input`, `os` import. Added `provider: Provider | None` param to `LLMJudge.__init__`; defaults to `AnthropicProvider(api_key=...)`. `assess()` now calls `self._provider.judge(model, system_prompt, messages, rubric_name)`. Judge prompts (`_JUDGE_SYSTEM_PROMPT`, `_JUDGE_USER_TEMPLATE`) remain in `judge.py` — they're application-level, not provider-specific.
- `SivoConfig` gains `provider: str = "anthropic"` and `judge_provider: str = ""` (empty = same as `provider`). `_parse_config` handles `[sivo] provider` and `[sivo.judge] provider`.
- All existing `patch("anthropic.AsyncAnthropic")` and `patch("anthropic.Anthropic")` mocks continue to work because the provider methods do `import anthropic` at call time, so the module-level patch still intercepts them.

---

## Performance tests (`tests/perf/`)

Excluded from the default `pytest` run. Run explicitly with `uv run pytest tests/perf/ -m perf`.

### Tasks (implement alongside Phase 9 or 10)

- [x] **Perf — Large JSONL throughput:** `tests/perf/test_large_jsonl.py`
  - Generate a synthetic JSONL fixture with 1,000 `ExecutionRecord` entries
  - Run `sivo replay` against it; assert completion in under 30 seconds
  - Assert peak RSS memory stays below 200 MB (use `tracemalloc`)

---

## Rename: evalkit → sivo

**Completed.** All 73 source files, tests, docs, and configuration files renamed from `evalkit` → `sivo` / `Evalkit` → `Sivo`. Specifically:

- `evalkit/` package directory → `sivo/`
- `pyproject.toml`: `name = "sivo"`, entry point `sivo = "sivo.cli:main"`
- All `from evalkit.X import ...` / `import evalkit` → `from sivo.X import ...` / `import sivo`
- `evalkit.toml` → `sivo.toml` in all config loading, docs, and tests
- `.evalkit` default store directory → `.sivo`
- `EvalkitConfig` → `SivoConfig`, `_evalkit_eval_` → `_sivo_eval_`
- Old `evalkit/` directory deleted; package reinstalled as `sivo==0.1.0`
- Full test suite: **520 passed, 7 skipped**

---

## Decisions log pointer

See `DECISIONS.md` for all architectural decisions and rationale. Do not re-litigate decisions recorded there.

---

## Blockers and open questions

_None currently. Add here as they arise._
