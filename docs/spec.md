# Sivo v1 — Product Specification

## 1. Philosophy

**The Unix tool for LLM evaluation.** `pip install sivo` and run. No dashboards, no SaaS accounts, no vendor lock-in.

Three principles drive every design decision:

- **Shift-left reliability.** Treat LLM evaluations exactly like unit tests. Catch prompt regressions and hallucinations in CI before they reach production.
- **Developer-first.** Optimise the local feedback loop — execution speed, zero-friction debugging, readable terminal output.
- **Reproducibility by default.** Separate LLM execution from evaluation. Every run produces a structured record that can be replayed without touching the API.

---

## 2. Scope

**v1 covers:** evaluation of single LLM calls and chat-style (multi-turn) inputs.

**Explicitly out of scope for v1:** multi-step agent execution, tool call inspection, step graphs. The data model supports these future extensions but the runner does not execute them.

---

## 3. Core Architecture

Sivo has two distinct layers. This separation is the key architectural decision.

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Execution Engine                          │
│  Calls the LLM. Produces ExecutionRecord.           │
└─────────────────────────────────────────────────────┘
                         │
                         ▼  JSONL store
┌─────────────────────────────────────────────────────┐
│  Layer 2: Eval Engine                               │
│  Consumes ExecutionRecord. Runs assertions.         │
│  Never calls the LLM directly.                      │
└─────────────────────────────────────────────────────┘
```

In **live mode** the execution engine calls the LLM and the eval engine runs immediately on the result.
In **replay mode** the execution engine is bypassed entirely — the eval engine loads stored records and re-runs assertions at zero API cost.

---

## 4. Data Model

### 4.1 EvalCase

The input contract for every eval function. The runner constructs this from an `ExecutionRecord`.

```python
class EvalCase:
    input: str | dict           # required — prompt or chat input
    output: str                 # required — model response to assert against

    system_prompt: str | None   # optional structured context
    conversation: list[Message] | None
    expected: Any | None        # optional ground truth
    metadata: dict              # run id, tags, dataset row, etc.
    tools: list | None          # reserved — not executed in v1
    trace: Trace | None         # reserved — read-only, not populated in v1
```

### 4.2 ExecutionRecord

The canonical stored artifact. Written by the execution engine, read by the eval engine and the replay runner.

```python
class ExecutionRecord:
    id: str                     # uuid
    timestamp: str              # ISO 8601
    run_id: str                 # groups records from one sivo run

    # Input
    input: Any
    system_prompt: str | None
    conversation: list[Message] | None

    # Output
    output: str

    # Full model provenance — exact strings, not aliases
    model: str                  # e.g. "claude-sonnet-4-6"
    params: dict                # temperature, max_tokens, etc.

    # Cost
    input_tokens: int
    output_tokens: int
    cost_usd: float

    # Metadata
    metadata: dict
    trace: Trace | None         # reserved — always None in v1
```

**Storage format:** JSONL (newline-delimited JSON). One record per line. Append-only.
Location: `.sivo/records/<run_id>.jsonl`

### 4.3 Trace (reserved)

```python
class Trace:
    steps: list[Step]           # defined but not populated in v1
```

Eval functions must not depend on this field being present. The runner will not populate it.

---

## 5. Eval Function Contract

### 5.1 Signature

```python
def eval_<name>(case: EvalCase) -> None:
    ...
```

- Functions named `eval_*` in files named `eval_*.py` are auto-discovered.
- Return value is ignored. Assertions raise on failure.
- The runner controls execution. The eval function defines assertions only.

### 5.2 The get_response pattern

Eval functions must retrieve the model output via the injected `get_response()` helper, not by calling the LLM directly. This is what makes replay work.

```python
# correct
def eval_refund_policy(case: EvalCase) -> None:
    response = case.output          # runner has already populated this
    assert_contains(response, "30 days")
    assert_judge(response, rubric="helpfulness")

# also correct — explicit helper form
def eval_refund_policy(case: EvalCase) -> None:
    response = get_response(case)   # injected by runner; returns case.output in replay
    assert_contains(response, "30 days")
```

In live mode `get_response()` returns `case.output` (already populated from the LLM call).
In replay mode `get_response()` returns the stored output. No API call is made.

### 5.3 Data-driven evals

To run an eval across a dataset, yield `EvalCase` objects from a companion function:

```python
def eval_refund_policy_cases():
    return [
        EvalCase(input="What is your refund policy?", expected="30 days"),
        EvalCase(input="Can I return after 60 days?",  expected="no"),
    ]

def eval_refund_policy(case: EvalCase) -> None:
    response = case.output
    assert_contains(response, case.expected)
```

The runner pairs the two functions by name prefix.

### 5.4 Fixtures

Shared setup (expensive clients, datasets, fixtures) is declared with `@sivo.fixture`:

```python
@sivo.fixture(scope="session")
def my_client():
    return AnthropicClient(...)

def eval_something(case: EvalCase, my_client) -> None:
    ...
```

Scope options: `"session"` (once per run), `"eval"` (once per eval function). Mirrors pytest fixture semantics deliberately.

---

## 6. Assertion Library

### 6.1 Deterministic assertions

Fast, free, no LLM call. Run first.

```python
assert_contains(text, substring)
assert_not_contains(text, substring)
assert_matches_schema(output, JsonSchema)
assert_regex(text, pattern)
assert_length(text, min=None, max=None)
```

### 6.2 LLM-as-judge: `assert_judge`

```python
assert_judge(
    output,
    rubric="helpfulness",              # built-in rubric name
    model="claude-haiku-4-5",          # default; configurable
)

assert_judge(
    output,
    rubric="The response must acknowledge the user's frustration before offering a solution.",
    model="ollama/llama3",             # local model for cost-sensitive suites
)
```

**Judge output schema (Pydantic):**
```python
class JudgeVerdict:
    passed: bool
    reason: str         # one sentence
    evidence: str       # quoted span from output that drove the verdict
    suggestion: str | None
```

All judge calls return structured JSON via Pydantic. No raw reasoning traces.

**Built-in rubrics (v1):** `helpfulness`, `tone`, `toxicity`, `factual_consistency`, `conciseness`.
Note: built-in rubrics are starting points, not ground truth. Known limitations are documented per rubric.

### 6.3 Assertion composition

Within a single eval function, all assertions must pass for the eval to pass. There is no weighting in v1. Evaluation stops at the first failure unless `--no-fail-fast` is set (see section 8).

### 6.4 Flakiness handling

LLM judge verdicts are non-deterministic. To avoid spurious CI failures, judge assertions run with a **retry budget**:

- Default: 1 attempt (standard mode)
- With `--reliable`: 3 attempts; verdict requires 2/3 majority
- Flaky results (e.g. 1 pass, 2 fail) are reported as `FLAKY` rather than `FAIL` and do not affect exit code by default. Set `--strict-flaky` to treat `FLAKY` as `FAIL`.

Deterministic assertions are never retried.

---

## 7. Execution Engine

- Async concurrency: independent LLM calls run in parallel via `asyncio`.
- Respects provider rate limits with configurable concurrency cap (`--concurrency N`, default 10).
- Built-in retry with exponential backoff for transient API errors (default: 3 attempts).
- Per-call timeout configurable (`--timeout N` seconds, default 30).
- Every call writes an `ExecutionRecord` to the JSONL store immediately on completion.

**Cost tracking:** token counts and USD cost are captured per record and aggregated at session level. Cost is attributed per eval function name for actionable reporting.

---

## 8. Terminal UI

### 8.1 Verbosity levels

| Flag | Output |
|------|--------|
| (default) | Pass/fail per eval + one-sentence reason on failure |
| `-v` | Expanded: failure context, evidence from judge |
| `-vv` | Debug: full `JudgeVerdict` JSON + exact prompts sent to model |

### 8.2 Session receipt

Printed at end of every run:

```
─────────────────────────────────────────
 sivo run complete
 12 passed  2 failed  1 flaky
 tokens: 48,210 in / 12,440 out
 cost:   $0.034  (eval_tone_check: $0.018, eval_factual: $0.012, other: $0.004)
 run id: run_20250114_143201
─────────────────────────────────────────
```

### 8.3 Execution modes

- `--fail-fast`: stop on first failure (default in interactive use)
- `--no-fail-fast`: run all evals, report at end (default in CI)
- `--concurrency N`: max parallel LLM calls
- `--reliable`: enable 3-attempt majority voting for judge assertions
- `--strict-flaky`: treat FLAKY as FAIL

### 8.4 Interactive REPL (`--pdb-llm`)

Pauses execution on a failing eval and drops into an interactive terminal session.

Available in v1:
- Inspect: `input`, `system_prompt`, `output`, `judge_verdict`
- Hot-swap: edit `system_prompt` and re-run the current eval in place
- Commands: `inspect`, `retry`, `skip`, `continue`, `abort`

Out of scope in v1 (future):
- Tool call inspection
- Step-by-step agent trace navigation

Hot-swapped prompts are not saved automatically. The dev copies changes back to their source manually. A future version may write patches to a diff file.

---

## 9. Replay System

### 9.1 CLI

```bash
sivo replay <run_id>                    # replay all records from a run
sivo replay <run_id> --eval eval_tone   # replay specific eval only
sivo replay <run_id> --filter model=claude-haiku-4-5
```

### 9.2 Behaviour

1. Load `ExecutionRecord` objects from `.sivo/records/<run_id>.jsonl`
2. Convert each record to an `EvalCase`
3. Run eval functions with `get_response()` returning `record.output`
4. No LLM calls made. Zero API cost.

### 9.3 Why this matters

- Write a new eval today → run it against last month's stored outputs immediately
- Change a rubric → verify it doesn't break existing passing cases
- Debug a CI failure → replay the exact execution locally with full context
- Cost: free after the initial run

---

## 10. CI Integration

### 10.1 Exit codes

| Code | Meaning |
|------|---------|
| 0 | All evals passed (FLAKY treated as pass unless `--strict-flaky`) |
| 1 | One or more evals FAILED |
| 2 | Sivo internal error (config, auth, timeout) |

### 10.2 Output format

Default CI output: **JUnit XML** (`--junit-xml <path>`). Parsed natively by GitHub Actions, Jenkins, CircleCI, and GitLab CI without glue code.

JSON summary also written to `.sivo/results/<run_id>.json` on every run.

---

## 11. Configuration

`sivo.toml` at project root:

```toml
[sivo]
default_model = "claude-haiku-4-5"
concurrency = 10
timeout = 30
store_path = ".sivo"

[sivo.judge]
default_model = "claude-haiku-4-5"
retry_attempts = 1            # set to 3 with --reliable

[sivo.cost]
warn_above_usd = 1.00         # warn if session cost exceeds this
```

---

## 12. Project Layout (Recommended)

```
my_agent/
├── sivo.toml
├── evals/
│   ├── eval_refund_policy.py
│   ├── eval_tone.py
│   └── fixtures.py
└── .sivo/
    ├── records/
    │   └── run_20250114_143201.jsonl
    └── results/
        └── run_20250114_143201.json
```

---

## 13. Out of Scope (v1)

The following are explicitly deferred and must not be implemented in v1:

- Multi-step agent execution and trace inspection
- Web UI or dashboard
- Remote result storage or team sharing
- Semantic diff between runs (planned v2)
- Automatic prompt patch suggestions from REPL
- LangGraph / LangChain adapters (post-v1)
