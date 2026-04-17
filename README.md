# sivo

**A pytest-style evaluation framework for LLM outputs.**

[![PyPI](https://img.shields.io/pypi/v/sivo)](https://pypi.org/project/sivo/)
[![CI](https://img.shields.io/github/actions/workflow/status/sivo-eval/sivo/ci.yml?branch=main)](https://github.com/sivo-eval/sivo/actions)
[![Python](https://img.shields.io/pypi/pyversions/sivo)](https://pypi.org/project/sivo/)
[![License](https://img.shields.io/github/license/sivo-eval/sivo)](LICENSE)

Write eval functions that assert on LLM outputs the way pytest asserts on function outputs. Every LLM call is stored as a JSONL record, so you can re-run new assertions against old outputs at zero API cost — and drop into an interactive REPL to fix failing prompts on the fly.

```python
from sivo.assertions import assert_contains, assert_judge

def eval_refund_policy(case):
    assert_contains(case.output, "30 days")
    assert_judge(case.output, rubric="tone")
```

```bash
pip install sivo
sivo run evals/
```

---

## Replay evals at zero API cost

![Replay demo](assets/demo_replay.gif)

Every LLM call writes an `ExecutionRecord` to JSONL. Write a new assertion today, run it against last month's outputs without touching the API. Change a rubric; verify it across your full history.

```bash
sivo replay run_20260101 evals/
```

## Debug failures interactively

![Interactive debug demo](assets/demo_pdb_llm.gif)

`--pdb-llm` pauses on every failure. Inspect the input, system prompt, and output. Hot-swap the prompt, hit `retry`, and sivo makes a fresh LLM call with your change. Find the fix in seconds instead of edit-run-wait loops.

```bash
sivo run evals/ --run-id my_run --pdb-llm
```

---

## Quick start

### Install

```bash
pip install sivo           # Anthropic (default)
pip install sivo[openai]   # + OpenAI support
```

Requires Python 3.11+. Set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`).

### Write your first eval

```python
# eval_support.py
from sivo.assertions import assert_contains, assert_judge

def eval_mentions_refund_window(case):
    """Response must state the 30-day return window."""
    assert_contains(case.output, "30 days")

def eval_tone(case):
    """Response must be professional in tone."""
    assert_judge(case.output, rubric="tone")
```

### Run it

```bash
# Replay against stored records (zero API cost)
sivo replay run_20260101 eval_support.py

# Or run data-driven evals (supply your own EvalCase objects)
sivo run eval_support.py
```

```
  PASS  eval_mentions_refund_window  (record r-001)
  FAIL  eval_tone  (record r-002)
        reason: Response uses casual language ("hey") instead of professional tone.

─────────────────────────────────────────
 sivo run complete
 1 passed  1 failed
 run id: run_20260101
─────────────────────────────────────────
```

---

See [`examples/`](examples/) for complete working evals including customer support bots and tone evaluation.

---

## What else

- **Multi-provider.** Anthropic and OpenAI built in. Execution and judge providers configured independently.
- **Structured LLM-as-judge.** Built-in rubrics (helpfulness, tone, toxicity, factual consistency, conciseness) or write your own in plain English. Verdicts come back as typed `JudgeVerdict` objects, cached by content hash.
- **CI-ready.** JUnit XML output, deterministic exit codes (`0` pass, `1` fail, `2` error), `--no-fail-fast` for complete reports.
- **Fixtures and data-driven evals.** pytest-style `@sivo.fixture(scope=...)` with session and eval scopes, plus `eval_*_cases()` generators for parametric evaluation.

---

## Core concepts

### Eval functions and the assertion model

An eval function is any `def eval_*(case: EvalCase)` in a file named `eval_*.py`. The function raises on failure;
returning normally is a pass. All assertions in a function must pass for the eval to pass.

```python
def eval_refund_policy(case):
    assert_contains(case.output, "30 days")
    assert_not_contains(case.output, "restocking fee")
    assert_judge(case.output, rubric="helpfulness")
```

Deterministic assertions (`assert_contains`, `assert_regex`, `assert_length`, `assert_matches_schema`) run free.
`assert_judge` makes an LLM call to the judge model and returns a structured `JudgeVerdict`.

### Two-layer architecture

```
Execution Engine  →  provider.complete()  →  ExecutionRecord (JSONL)
Eval Engine       →  reads JSONL          →  assertions       →  pass/fail
```

Eval functions never call the LLM. The runner injects a pre-populated `EvalCase`; `case.output` holds the stored
response. This is what makes replay work: the eval engine is bypassed by the execution layer at run time, and
re-run against stored records at zero cost afterward.

### LLM judge

```python
# Built-in rubrics: helpfulness, tone, toxicity, factual_consistency, conciseness
assert_judge(response, rubric="tone")

# Custom rubric — plain English
assert_judge(
    response,
    rubric="The response must acknowledge the customer's frustration before offering a solution.",
)
```

Judge calls return a structured `JudgeVerdict(passed, reason, evidence, suggestion)`. Results are cached by
content hash — identical (rubric, output) pairs never hit the API twice within a session.

### Fixtures and data-driven evals

```python
import sivo
from sivo.models import EvalCase

@sivo.fixture(scope="session")   # "eval" scope also available
def db_client():
    client = MyDBClient(url=os.environ["DB_URL"])
    yield client
    client.close()

# Data-driven: pair eval_*_cases() with eval_*()
def eval_sentiment_cases():
    return [
        EvalCase(input="Great product!", output="positive sentiment detected"),
        EvalCase(input="Terrible service.", output="negative sentiment detected"),
    ]

def eval_sentiment(case, db_client):
    assert_contains(case.output, case.expected)
```

No `--run-id` required for data-driven evals. Fixture scoping mirrors pytest: `"session"` initialises once per run,
`"eval"` once per eval function.

## Multi-provider support

Anthropic (default) and OpenAI are built in. Execution provider and judge provider are independent.

```bash
sivo run evals/ --run-id my_run --provider openai --judge-provider anthropic
sivo run evals/ --run-id my_run --judge-provider openai --judge-model gpt-4o-mini
```

```toml
# sivo.toml
[sivo]
provider = "anthropic"

[sivo.judge]
provider = "openai"
default_model = "gpt-4o-mini"
```

| Provider | Install | Env var |
|---|---|---|
| `anthropic` (default) | included | `ANTHROPIC_API_KEY` |
| `openai` | `pip install sivo[openai]` | `OPENAI_API_KEY` |

Custom providers implement the `Provider` protocol (`sivo.providers.Provider`) and are loaded by import path:
`--judge-provider "my_pkg.module:MyProvider"`. See [docs/MULTI_LLM_SPEC.md](docs/MULTI_LLM_SPEC.md).

---

## CLI reference

| Command | Description |
|---|---|
| `sivo run [path]` | Discover and run eval functions |
| `sivo replay RUN_ID [path]` | Replay stored records through evals (no LLM calls) |

| Flag | Applies to | Description |
|---|---|---|
| `--run-id RUN_ID` | `run` | Load records from this run ID |
| `--eval NAME` | both | Run only this eval function |
| `--filter KEY=VALUE` | `replay` | Filter records by metadata (repeatable) |
| `--no-fail-fast` | both | Run all evals; don't stop on first failure |
| `--store-path PATH` | both | Data store root (default: `.sivo`) |
| `-v` / `-vv` | both | Failure evidence / full judge JSON |
| `--pdb-llm` | `run` | Interactive REPL on every failure |
| `--junit-xml PATH` | both | Write JUnit XML report |
| `--strict-flaky` | both | Treat `FLAKY` as `FAIL` (exit 1) |
| `--provider PROVIDER` | both | Execution provider (`anthropic`, `openai`, or import path) |
| `--judge-provider PROVIDER` | both | Judge provider (defaults to `--provider`) |
| `--judge-model MODEL` | both | Judge model (default: `claude-haiku-4-5`) |

Exit codes: `0` = all pass, `1` = any fail, `2` = internal error.

## Configuration

```toml
# sivo.toml (searched upward from cwd — file is optional)

[sivo]
default_model = "claude-haiku-4-5"
concurrency   = 10
timeout       = 30
store_path    = ".sivo"
provider      = "anthropic"

[sivo.judge]
default_model   = "claude-haiku-4-5"
provider        = ""
retry_attempts  = 1

[sivo.cost]
warn_above_usd = 1.00
```

---

## CI integration

```yaml
# .github/workflows/evals.yml
- name: Run evals
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: |
    sivo replay ${{ env.RUN_ID }} evals/ \
      --no-fail-fast \
      --junit-xml eval-results.xml

- uses: mikepenz/action-junit-report@v4
  if: always()
  with:
    report_paths: eval-results.xml
```

A JSON summary is written automatically to `.sivo/results/<run_id>.json` on every run.

---

## Contributing

Open an issue or PR at [github.com/sivo-eval/sivo](https://github.com/sivo-eval/sivo).
Bug reports should include a minimal reproducible eval file and the `run_id` (or a canned JSONL fixture).

---

## License

MIT
