# sivo

**Developer-first LLM evaluation. Think pytest for LLMs.**

`pip install sivo`, write `eval_*.py` files, run `sivo run`. No dashboards, no SaaS, no vendor lock-in.

---

## Why sivo?

LLM outputs are non-deterministic. Prompts break silently. Sivo treats LLM evaluation exactly like unit tests — catch regressions in CI before they reach production.

The key design: **separate execution from evaluation**. Every LLM call produces a stored `ExecutionRecord`. Eval functions run assertions against stored records — at zero API cost — forever.

```
LLM call → ExecutionRecord (JSONL) → eval functions → pass/fail
                                      ↑
                              replay any time, free
```

---

## Install

```bash
pip install sivo           # Anthropic (default)
pip install sivo[openai]   # + OpenAI support
```

Requires Python 3.11+. Set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` for OpenAI) to use the execution engine and LLM judge.

---

## Quickstart

**1. Write an eval file:**

```python
# eval_refund_policy.py
from sivo.assertions import assert_contains, assert_judge

def eval_mentions_30_days(case):
    assert_contains(case.output, "30 days")

def eval_tone(case):
    assert_judge(case.output, rubric="tone")
```

**2. Run evals against stored records:**

```bash
sivo replay <run_id> eval_refund_policy.py
```

**3. Or use data-driven evals (no stored records needed):**

```python
# eval_suite.py
from sivo.models import EvalCase
from sivo.assertions import assert_contains

def eval_refund_cases():
    return [
        EvalCase(input="Refund policy?", output="Returns accepted within 30 days."),
        EvalCase(input="Return after 60 days?", output="Contact support for exceptions."),
    ]

def eval_refund(case):
    assert_contains(case.output, "support")
```

```bash
sivo run eval_suite.py   # no --run-id needed for data-driven evals
```

---

## How it works

### Two-layer architecture

```
Layer 1: Execution Engine   →  calls LLM  →  writes ExecutionRecord to JSONL
Layer 2: Eval Engine        →  reads JSONL →  runs assertions  →  pass/fail
```

Eval functions **never call the LLM**. The runner injects pre-populated `EvalCase` objects. This is what makes replay work.

### Eval function contract

```python
def eval_<name>(case: EvalCase) -> None:
    response = case.output   # already populated by the runner
    assert_contains(response, "expected text")
```

- Named `eval_*` in files named `eval_*.py` — auto-discovered
- Return value ignored; raise to fail
- Runner controls execution; eval function defines assertions only

---

## Assertion library

### Deterministic (free, no API call)

```python
from sivo.assertions import (
    assert_contains,       # substring match
    assert_not_contains,   # negative match
    assert_regex,          # regex pattern
    assert_length,         # min/max character count
    assert_matches_schema, # Pydantic model or TypeAdapter-compatible type
)

assert_contains(response, "30 days")
assert_not_contains(response, "restocking fee")
assert_regex(response, r"\d+ days?")
assert_length(response, min=10, max=500)
assert_matches_schema(response, MyResponseModel)
```

### LLM judge (makes API call)

```python
from sivo.judge import assert_judge

# Built-in rubrics: helpfulness, tone, toxicity, factual_consistency, conciseness
assert_judge(response, rubric="tone")

# Custom rubric
assert_judge(
    response,
    rubric="The response must acknowledge frustration before offering a solution.",
    model="claude-haiku-4-5",  # default
)
```

Judge calls are cached by content hash — identical (rubric, output) pairs never hit the API twice.

---

## Fixtures

Share expensive setup across evals using `@sivo.fixture`:

```python
import sivo

@sivo.fixture(scope="session")   # initialised once per run
def db_client():
    client = MyDBClient(url=os.environ["DB_URL"])
    yield client
    client.close()   # teardown runs after the session

def eval_db_roundtrip(case, db_client):
    result = db_client.query(case.input)
    assert_contains(result, case.expected)
```

Scope options:
- `"session"` — initialised once, shared across all evals
- `"eval"` — initialised once per eval function, reset between them

---

## Data-driven evals

Pair an `eval_*_cases()` function with an `eval_*()` function to run one eval per case:

```python
from sivo.models import EvalCase

def eval_sentiment_cases():
    return [
        EvalCase(input="Great product!", output="positive"),
        EvalCase(input="Terrible service.", output="negative"),
    ]

def eval_sentiment(case):
    assert_contains(case.output, case.expected)
```

No `--run-id` or store required. Case IDs are auto-assigned as `case-0`, `case-1`, …

---

## CLI reference

### `sivo run`

Discover and run eval functions. For data-driven evals: no `--run-id` needed.
For record-based evals: requires `--run-id`.

```bash
sivo run [path] [--run-id RUN_ID] [--eval NAME] [--no-fail-fast]
            [--store-path PATH] [-v|-vv] [--pdb-llm]
            [--junit-xml PATH] [--strict-flaky]
```

### `sivo replay`

Replay stored records through eval functions — zero API cost.

```bash
sivo replay RUN_ID [path] [--eval NAME] [--filter KEY=VALUE]
               [--no-fail-fast] [--store-path PATH] [-v|-vv]
               [--junit-xml PATH] [--strict-flaky]
```

### Options

| Flag | Description |
|------|-------------|
| `--run-id RUN_ID` | Load records from this run (required for non-data-driven evals) |
| `--eval NAME` | Run only the named eval function |
| `--filter KEY=VALUE` | Filter records by metadata field (repeatable) |
| `--no-fail-fast` | Continue after failures (default: stop on first) |
| `--store-path PATH` | Data store root (default: `.sivo`) |
| `-v` / `-vv` | Verbosity: failure evidence / full judge JSON |
| `--pdb-llm` | Interactive REPL on every failure |
| `--junit-xml PATH` | Write JUnit XML (CI integration) |
| `--strict-flaky` | Treat FLAKY as FAIL (exit code 1) |
| `--provider PROVIDER` | LLM provider for execution (`anthropic`, `openai`, or import path) |
| `--judge-provider PROVIDER` | LLM provider for the judge (defaults to `--provider`) |
| `--judge-model MODEL` | Model for the judge (default: `claude-haiku-4-5`) |

---

## CI integration

```yaml
# .github/workflows/eval.yml
- name: Run evals
  run: |
    sivo replay ${{ env.RUN_ID }} evals/ \
      --no-fail-fast \
      --junit-xml eval-results.xml

- uses: mikepenz/action-junit-report@v4
  with:
    report_paths: eval-results.xml
```

Exit codes: `0` = all pass, `1` = any fail, `2` = error.

JSON summary is written automatically to `.sivo/results/<run_id>.json`.

---

## Configuration — `sivo.toml`

Optional project-level config file:

```toml
[sivo]
default_model = "claude-haiku-4-5"
concurrency = 10
timeout = 30
store_path = ".sivo"

[sivo.judge]
default_model = "claude-haiku-4-5"
retry_attempts = 1      # set to 3 to enable majority-vote flakiness handling

[sivo.cost]
warn_above_usd = 1.00   # print a warning if session cost exceeds this
```

sivo searches the current directory and all ancestor directories for `sivo.toml`.

---

## Multi-provider support

sivo ships with Anthropic (default) and OpenAI providers. The execution engine and LLM judge can use any provider independently.

### Switching providers

**CLI flags:**

```bash
# Use OpenAI as the judge provider
sivo run evals/ --run-id my_run --judge-provider openai --judge-model gpt-4o-mini

# Override both the execution provider and the judge provider
sivo run evals/ --run-id my_run --provider openai --judge-provider anthropic
```

**sivo.toml:**

```toml
[sivo]
provider = "anthropic"         # execution provider (default: "anthropic")

[sivo.judge]
provider = "openai"            # judge can use a different provider
default_model = "gpt-4o-mini"  # override the judge model
```

### Built-in providers

| Provider | Install | API key env var |
|----------|---------|-----------------|
| `anthropic` (default) | included | `ANTHROPIC_API_KEY` |
| `openai` | `pip install sivo[openai]` | `OPENAI_API_KEY` |

### Custom providers

Implement the `Provider` protocol and reference it by import path:

```python
# my_project/my_provider.py
from sivo.providers import CompletionResult
from sivo.models import JudgeVerdict

class MyProvider:
    name = "my_provider"

    async def complete(self, *, model, system_prompt, messages,
                       max_tokens=1024, timeout=30.0, extra_params=None):
        ...
        return CompletionResult(output=..., input_tokens=...,
                                output_tokens=..., cost_usd=..., model=model)

    def judge(self, *, model, system_prompt, messages, rubric_name):
        ...
        return JudgeVerdict(passed=..., reason=..., evidence=...)
```

```toml
# sivo.toml
[sivo.judge]
provider = "my_project.my_provider:MyProvider"
```

Or via CLI:

```bash
sivo run evals/ --run-id my_run \
  --judge-provider "my_project.my_provider:MyProvider"
```

---

## Interactive REPL — `--pdb-llm`

Pause on every failing eval to inspect context and hot-swap the system prompt:

```
sivo run evals/ --run-id run_20260101 --pdb-llm
```

REPL commands:
- `inspect` — show `input`, `system_prompt`, `output`, `judge_verdict`
- `retry` — re-run the current eval after hot-swapping
- `system_prompt = "new prompt"` — hot-swap the system prompt
- `skip` — mark as skipped and continue
- `continue` — disable the REPL and finish the run
- `abort` — stop immediately

---

## Project layout

```
my_project/
├── sivo.toml           # optional config
├── evals/
│   ├── eval_tone.py       # eval functions
│   └── eval_accuracy.py
└── .sivo/
    ├── records/
    │   └── run_20260101.jsonl   # stored ExecutionRecords
    └── results/
        └── run_20260101.json    # JSON summary (auto-written)
```

---

## License

MIT
