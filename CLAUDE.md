# CLAUDE.md — Sivo

Read this file, `PROGRESS.md`, and `docs/spec.md` before doing anything else in a new session.

---

## What this project is

Sivo is a developer-first Python library for evaluating LLM outputs. Think pytest for LLMs. `pip install sivo`, write `eval_*.py` files, run `sivo run`. No dashboards, no SaaS, no vendor lock-in.

The core idea: separate LLM execution from evaluation so that evals can be replayed against stored outputs at zero API cost.

---

## Tech stack

- **Python 3.11+**
- **uv** for package management (not pip, not poetry)
- **Pydantic v2** for all data models
- **asyncio** for concurrent LLM execution
- **Rich** for terminal UI (progress bars, tables, REPL)
- **pytest** as the test framework for sivo's own tests
- **JSONL** for the local result store (append-only, one record per line)
- **JUnit XML** for CI output

Anthropic SDK is the primary LLM client for v1. The judge model defaults to `claude-haiku-4-5`.

API keys e.g. ANTHROPIC API KEY is loaded from .envrc via python-dotenv

---

## Architecture — the two most important things

**1. Two-layer separation (non-negotiable)**

```
Execution Engine  →  produces ExecutionRecord  →  written to JSONL
Eval Engine       →  consumes ExecutionRecord  →  runs assertions
```

The eval engine never calls the LLM directly. It only reads from `ExecutionRecord`. This is what makes replay work.

**2. The `get_response()` injection pattern**

Eval functions retrieve output via `case.output` or the `get_response(case)` helper. The runner injects the correct behaviour — live or replay — transparently. Eval functions must never call the LLM themselves.

```python
def eval_something(case: EvalCase) -> None:
    response = case.output      # runner has already populated this
    assert_contains(response, "expected text")
    assert_judge(response, rubric="helpfulness")
```

---

## Key constraints for v1

- Single LLM call evaluation only — no multi-step agent execution
- `trace` field on `EvalCase` is reserved and read-only — do not populate it
- No web UI, no remote storage, no team sharing features
- Assertions within an eval are all-or-nothing — no weighting
- The runner controls execution; eval functions define assertions only

---

## Project layout

```
sivo/
├── CLAUDE.md               ← you are here
├── PROGRESS.md             ← read this every session
├── DECISIONS.md            ← architectural decisions log
├── docs/
│   └── spec.md             ← full v1 specification
├── pyproject.toml
├── sivo/
│   ├── __init__.py
│   ├── models.py           ← EvalCase, ExecutionRecord, JudgeVerdict, Trace
│   ├── runner.py           ← ExecutionEngine, EvalEngine, orchestration
│   ├── assertions.py       ← assert_contains, assert_judge, etc.
│   ├── judge.py            ← LLMJudge, built-in rubrics, caching
│   ├── discovery.py        ← auto-discovery of eval_* functions
│   ├── fixtures.py         ← @sivo.fixture decorator
│   ├── store.py            ← JSONL read/write, ExecutionRecord persistence
│   ├── replay.py           ← replay runner
│   ├── repl.py             ← --pdb-llm interactive REPL
│   ├── report.py           ← terminal UI, session receipt, JUnit XML
│   └── cli.py              ← sivo run / replay / inspect commands
├── tests/
│   ├── test_models.py
│   ├── test_assertions.py
│   ├── test_runner.py
│   ├── test_store.py
│   └── test_replay.py
└── examples/
    ├── eval_refund_policy.py
    └── eval_tone.py
```

---

## Session discipline

- Always read `PROGRESS.md` before starting work
- Always run `uv run pytest` before ending a session — do not leave tests failing
- Do not implement anything marked out of scope in `DECISIONS.md` or the spec
- When making a new architectural decision, note it in `DECISIONS.md`
- Ask before adding new dependencies
