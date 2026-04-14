# DECISIONS.md — Sivo

A log of architectural and product decisions. Includes context and rationale so decisions are not relitigated.

If you are a Claude Code session: do not contradict or work around decisions recorded here. If a decision needs revisiting, flag it to the developer rather than unilaterally changing it.

---

## D-001 — Two-layer execution/eval separation

**Decision:** Split the system into a distinct Execution Engine (calls LLM, produces `ExecutionRecord`) and Eval Engine (consumes records, runs assertions). The eval engine never calls the LLM.

**Rationale:** This is the architectural decision that makes replay possible. Storing `ExecutionRecord` as the canonical artifact means new evals can be run against old data at zero API cost. It also makes the system easier to test — the eval engine can be tested with fully mocked records.

**Implications:** All eval functions must retrieve output via `case.output` or `get_response(case)`. They must never call the LLM directly.

---

## D-002 — JSONL as the storage format

**Decision:** Store `ExecutionRecord` objects as newline-delimited JSON (JSONL), one record per line, in `.sivo/records/<run_id>.jsonl`.

**Rationale:** Append-only, human-readable, trivially greppable from the CLI, easy to diff between runs, no database dependency. SQLite was considered but adds complexity with no meaningful benefit for v1 access patterns.

**Implications:** No random-access updates to records. If a record needs amending it is appended as a new entry. The replay runner reads all records for a given `run_id` and filters in memory.

---

## D-003 — v1 scope: single-call evaluation only

**Decision:** v1 evaluates single LLM calls and chat-style (multi-turn) inputs only. Multi-step agent execution, tool call inspection, and step graphs are explicitly out of scope.

**Rationale:** Agent evaluation explodes execution model complexity and makes replay significantly harder. The `EvalCase.trace` and `EvalCase.tools` fields are defined in the data model for future compatibility but are not populated by the runner.

**Implications:** The `--pdb-llm` REPL in v1 exposes `input`, `system_prompt`, `output`, and `judge_verdict` only. Tool call navigation is a v2 feature.

---

## D-004 — Eval function contract: assert-style, no return value

**Decision:** Eval functions take `EvalCase` and return `None`. Assertions raise on failure. Return values are ignored.

**Alternatives considered:**
- `def eval_x(case) -> bool` — rejected, too limiting, no structured failure context
- `def eval_x(case) -> EvalResult` — rejected, moves too much responsibility into the function, breaks the runner-controls-execution principle

**Rationale:** Mirrors pytest semantics exactly. Any pytest user will understand it immediately. Structured failure context is carried by the `AssertionError` subclass, not the return value.

---

## D-005 — Runner controls execution, eval defines assertions

**Decision:** The runner is responsible for calling the LLM, constructing `EvalCase`, injecting `get_response()`, and handling retries. Eval functions are pure assertion logic only.

**Rationale:** Keeps eval functions simple and portable. Enables replay (runner swaps in stored output transparently). Prevents eval functions from accumulating execution concerns over time.

**Implications:** If an eval function needs a live LLM call inside it (e.g. to generate a second response for comparison), this is not supported in v1. The eval engine has one input and one output per case.

---

## D-006 — LLM judge uses Pydantic structured outputs only

**Decision:** `LLMJudge` returns a `JudgeVerdict` Pydantic model. No raw reasoning traces, no unstructured text parsing.

**Rationale:** Structured outputs are reliable to parse, testable, and composable. Raw traces vary by model and are difficult to act on programmatically. The `reason`, `evidence`, and `suggestion` fields cover the useful parts of a reasoning trace without exposing implementation details.

---

## D-007 — Judge caching by content hash

**Decision:** LLM judge calls are cached by `SHA256(rubric + output)`. Same rubric and output → same verdict returned from cache, no API call made.

**Rationale:** Judge calls on identical content in replay or repeated runs are wasteful. Deterministic caching is safe here because the judge prompt is fully determined by rubric + output.

**Implications:** Cache is session-scoped in v1 (in-memory). A persistent cache across sessions is a v2 feature.

---

## D-008 — Flakiness: FLAKY status with opt-in strictness

**Decision:** Judge assertions default to 1 attempt. `--reliable` enables 3-attempt majority voting. Results that split (e.g. 2 pass, 1 fail) are reported as `FLAKY` and do not fail CI by default. `--strict-flaky` makes `FLAKY` fail CI.

**Rationale:** LLM non-determinism is real. A binary pass/fail with no flakiness handling would cause spurious CI failures and erode trust in the test suite. Making flakiness visible but non-blocking by default is the pragmatic choice for v1.

**Implications:** The default `--reliable` budget is 3 attempts (2/3 majority). This triples judge costs when used. Should be used selectively.

---

## D-009 — JUnit XML as CI output format

**Decision:** CI-mode output is JUnit XML via `--junit-xml <path>`. JSON summary is also written automatically to `.sivo/results/<run_id>.json`.

**Rationale:** JUnit XML is parsed natively by GitHub Actions, Jenkins, CircleCI, and GitLab CI with no glue code. It is the de facto standard. TAP was considered but has weaker tooling support.

---

## D-010 — Fixtures mirror pytest scope model

**Decision:** `@sivo.fixture(scope="session"|"eval")` for shared setup. Session-scoped fixtures initialise once per run; eval-scoped fixtures initialise once per eval function.

**Rationale:** Pytest's fixture model is well understood. Mirroring it reduces cognitive load. Function-scope fixtures (per data-driven case) are deferred to v2 to keep the v1 implementation tractable.

---

## D-011 — Assertion composition: all-or-nothing, stop on first failure

**Decision:** All assertions within an eval function must pass for the eval to pass. No weighting. Execution stops at first assertion failure unless `--no-fail-fast` is set at the session level.

**Rationale:** Weighting adds complexity with unclear benefit for v1. All-or-nothing is predictable and mirrors standard test behaviour. Per-eval fail-fast control (rather than session-level) is a v2 consideration.

---

## D-012 — Model version must be captured as exact string

**Decision:** `ExecutionRecord.model` stores the exact model string (e.g. `"claude-haiku-4-5"` not `"claude"` or `"haiku"`). This is required.

**Rationale:** Model updates are the most common source of silent regressions in LLM systems. Vague model aliases make it impossible to reproduce a stored execution. Exact strings are mandatory for the replay system to be trustworthy.

---

## D-013 — uv for package management

**Decision:** Use `uv` for all package management. Not pip, not poetry, not conda.

**Rationale:** Speed, lockfile reliability, and modern Python packaging standards. Consistent with the developer-first positioning of the tool.

---

## D-014 — LLM provider abstraction (resolved — Multi-LLM phases A–C)

**Original decision (Phase 10):** v1 ships Anthropic-only. All provider-specific code must be isolated so that multi-provider support can be added cleanly in a future phase, but no abstraction layer is introduced now.

**Resolution (Multi-LLM phases A–C, 2026-04-14):** Multi-provider support is now fully implemented.

- **Phase A** — Extracted all Anthropic-specific code into `sivo/providers/anthropic.py`. Introduced `Provider` protocol (`sivo/providers/__init__.py`) and `get_provider()` registry (`sivo/providers/registry.py`). Wired `ExecutionEngine` and `LLMJudge` to use provider instances. Zero behaviour change to existing tests.
- **Phase B** — Implemented `OpenAIProvider` (`sivo/providers/openai.py`) with function-calling judge and a prompt-based JSON fallback utility (`sivo/providers/_fallback_judge.py`). Added `openai` as an optional dependency (`pip install sivo[openai]`).
- **Phase C** — Wired `--provider`, `--judge-provider`, `--judge-model` CLI flags (both `run` and `replay`). `sivo.toml` `[sivo] provider` and `[sivo.judge] provider` are now honoured. Updated judge cache key to include model (D-007 update). Session judge override pattern (`set_session_judge()`) allows `run_session` to inject a custom judge without leaking module state.

**Architecture summary:**
- `Provider` is a `@runtime_checkable Protocol` — no inheritance required
- `get_provider(name)` resolves `"anthropic"` / `"openai"` (lazy for optional deps) and `"my_pkg.mod:ClassName"` (custom)
- Judge provider is independent of the execution provider — can mix and match
- Cache key is now `SHA256(model + rubric + output)` — prevents false hits across providers

---

### Phase 10 provider isolation audit

Conducted in Phase 10 (2026-04-13). All Anthropic-specific code remains in the two leaf methods identified below. No provider coupling has leaked into orchestration, session, or fixture code added in Phases 4–9.

**Isolation is confirmed adequate for v1.** A future multi-provider refactor would only touch:
1. `ExecutionEngine._call_llm()` in `runner.py` — replace with a provider-agnostic `_call_provider()` that accepts a `LLMProvider` protocol.
2. `LLMJudge._call_judge()` and `LLMJudge._extract_tool_input()` in `judge.py` — replace the Anthropic tool-use pattern with a provider-generic structured-output protocol.
3. `_COST_TABLE` in `runner.py` — extend with per-provider cost tables or move cost tracking into the provider adapter.

No changes to models, store, fixtures, discovery, runner orchestration, CLI, or report are needed for a provider refactor.

---

### Anthropic-specific code inventory (updated Phase 10)

All locations where provider code is currently coupled to Anthropic. This is the complete scope of work for a future multi-provider refactor.

**`sivo/judge.py`** — all provider code in `_call_judge()` and `_extract_tool_input()`

| Lines | What is Anthropic-specific |
|-------|---------------------------|
| 184 | `import anthropic` — deferred SDK import (inside `_call_judge`) |
| 186 | `anthropic.Anthropic(api_key=...)` — synchronous client instantiation |
| 188–203 | `client.messages.create(model=, max_tokens=, system=, tools=, tool_choice=, messages=)` — Anthropic Messages API call shape |
| 194 | `tool_choice={"type": "tool", "name": "record_verdict"}` — Anthropic-specific forced-tool-call syntax for structured output |
| 79–113 | `_build_judge_tool()` returns an Anthropic tool definition dict (`name`, `description`, `input_schema`) — not portable to OpenAI function-calling format |
| 213 | `block.type == "tool_use"` — Anthropic response block type name |
| 213 | `block.name == "record_verdict"` — Anthropic `tool_use` block attribute |
| 214 | `block.input` — Anthropic `tool_use` input dict (OpenAI uses `function.arguments` as a JSON string) |

**`sivo/runner.py`** — all provider code in `ExecutionEngine._call_llm()`

| Lines | What is Anthropic-specific |
|-------|---------------------------|
| 189 | `import anthropic` — deferred SDK import (inside `_call_llm`) |
| 191 | `anthropic.AsyncAnthropic(api_key=...)` — async client instantiation |
| 193–208 | `client.messages.create(...)` — Anthropic Messages API call shape |
| 209 | `response.content[0].text` — Anthropic response content block extraction |
| 210–211 | `response.usage.input_tokens` / `response.usage.output_tokens` — Anthropic usage field names |
| 30–42 | `_COST_TABLE` and `_calculate_cost()` — Anthropic model name strings and USD pricing |

**`sivo/models.py`**

| Lines | What is Anthropic-specific |
|-------|---------------------------|
| 107–108 | `input_tokens` / `output_tokens` field names on `ExecutionRecord` — mirrors Anthropic's usage schema |

---

## Open questions

_None currently. Add here as they arise during development._
