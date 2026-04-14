# Multi-LLM Provider Support — Spec

**Status:** Draft
**Scope:** Add a provider abstraction layer so Sivo works with multiple LLM providers. Ship with Anthropic + OpenAI; expose a clear `Provider` protocol for third-party extension.

---

## Context

Decision 14 confirmed all Anthropic-specific code is isolated to two locations:

- `_call_llm` in the execution engine (makes completion calls, extracts tokens/cost)
- `_call_judge` / `_extract_tool_input` in the judge (uses Anthropic tool_use for structured verdicts)

This spec defines how to abstract those touchpoints behind a provider protocol without disrupting the existing architecture.

---

## Design principles

1. **Zero breaking changes** — existing Anthropic-only usage must work identically without config changes
2. **Provider as a protocol** — use `typing.Protocol` so third parties can implement providers without inheriting from a base class
3. **Two providers ship** — `AnthropicProvider` and `OpenAIProvider`; others are community-contributed
4. **Provider handles its own format** — message construction, response parsing, token extraction, and cost calculation are the provider's responsibility
5. **Judge portability** — structured output must work across providers using each provider's native mechanism (tool_use for Anthropic, function calling for OpenAI, prompt-based JSON fallback for others)

---

## Provider protocol

```python
from typing import Protocol, runtime_checkable
from sivo.models import ExecutionRecord, JudgeVerdict

@runtime_checkable
class Provider(Protocol):
    """Interface that every LLM provider must implement."""

    name: str  # e.g. "anthropic", "openai"

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 30.0,
    ) -> CompletionResult:
        """Make an LLM completion call and return a normalised result."""
        ...

    async def judge(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[dict],
        rubric_name: str,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> JudgeVerdict:
        """Make a structured judge call and return a JudgeVerdict."""
        ...
```

### CompletionResult

A normalised response container returned by `complete()`:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class CompletionResult:
    output: str                    # The LLM's text response
    input_tokens: int              # Tokens consumed by the input
    output_tokens: int             # Tokens in the response
    cost_usd: float                # Estimated cost for this call
    model: str                     # Model string actually used
    raw_response: dict | None      # Provider-specific raw response (for debugging)
```

---

## Provider implementations

### AnthropicProvider

Wraps the existing `_call_llm` and `_call_judge` logic. This is a refactor, not a rewrite — lift the current Anthropic-specific code into the provider class.

- `complete()` → uses `anthropic.AsyncAnthropic().messages.create()`
- `judge()` → uses tool_use (`record_verdict` tool) as today
- Cost calculation: uses Anthropic's published per-model pricing (hardcoded table, overridable via config)

### OpenAIProvider

New implementation targeting the OpenAI API.

- `complete()` → uses `openai.AsyncOpenAI().chat.completions.create()`
- `judge()` → uses function calling (`functions` parameter) to extract structured verdicts
- Message format mapping: Sivo's `messages: list[dict]` maps directly to OpenAI's format (both use `role` + `content`)
- Cost calculation: uses OpenAI's published per-model pricing (hardcoded table, overridable via config)

### Fallback judge strategy

For providers that don't support tool_use or function calling natively, the judge should fall back to:

1. Append a JSON schema to the system prompt describing the expected `JudgeVerdict` shape
2. Ask for a raw JSON response
3. Parse and validate with Pydantic

This fallback lives in a shared utility so future providers can reuse it.

---

## Configuration

### sivo.toml changes

```toml
[sivo]
provider = "anthropic"           # "anthropic" | "openai" | "custom.module:MyProvider"
model = "claude-sonnet-4-20250514"

# Provider-specific config sections
[sivo.anthropic]
api_key_env = "ANTHROPIC_API_KEY"  # env var name (never store keys in config)

[sivo.openai]
api_key_env = "OPENAI_API_KEY"
base_url = ""                      # optional, for Azure OpenAI or proxies

[sivo.judge]
provider = "anthropic"             # judge can use a different provider than execution
model = "claude-sonnet-4-20250514"
```

### Key design decisions

- **`provider` is a string** — built-in values are `"anthropic"` and `"openai"`. For custom providers, use a dotted import path: `"my_package.providers:MyProvider"`
- **Judge provider is independent** — you might execute with OpenAI but judge with Anthropic (or vice versa). Defaults to the top-level provider if not specified.
- **API keys come from env vars** — the config only stores the env var name, never the key itself. This is safe for version control.
- **Defaults preserve current behaviour** — if no provider is specified, Sivo defaults to `"anthropic"` with the existing model defaults.

### CLI changes

```bash
# Override provider for a single run
sivo run --provider openai --model gpt-4o evals/

# Override judge provider independently
sivo run --judge-provider anthropic --judge-model claude-sonnet-4-20250514 evals/
```

---

## Provider registry

A simple registry that maps provider names to classes:

```python
_BUILTIN_PROVIDERS: dict[str, type[Provider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}

def get_provider(name: str, config: dict) -> Provider:
    """Resolve a provider by name or import path."""
    if name in _BUILTIN_PROVIDERS:
        return _BUILTIN_PROVIDERS[name](config)
    # Custom provider: "my_package.module:ClassName"
    module_path, class_name = name.rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not isinstance(cls, type) or not issubclass(cls, Provider):
        raise TypeError(f"{name} does not implement the Provider protocol")
    return cls(config)
```

---

## Integration points

### Execution engine

The execution engine currently calls `_call_llm` directly. Change to:

```python
# Before
record = await self._call_llm(case, ...)

# After
result = await self.provider.complete(
    model=self.config.model,
    system_prompt=case.system_prompt,
    messages=case.messages,
    ...
)
record = ExecutionRecord.from_completion_result(case, result)
```

### Judge

The judge currently uses `_call_judge` with Anthropic tool_use. Change to:

```python
# Before
verdict = await self._call_judge(output, rubric, ...)

# After
verdict = await self.judge_provider.judge(
    model=self.config.judge_model,
    system_prompt=self._build_judge_prompt(rubric),
    messages=[{"role": "user", "content": output}],
    rubric_name=rubric.name,
    ...
)
```

### Cache compatibility

The existing judge cache uses SHA256 of (rubric + output). Update to include the model in the cache key — SHA256 of (model + rubric + output). This prevents false cache hits when comparing providers or switching models, at the cost of no cache reuse across providers for identical rubric+output pairs. This is the safer default.

---

## Implementation phases

### Phase A — Provider protocol + registry (1 session)

- [ ] Create `sivo/providers/__init__.py` with `Provider` protocol and `CompletionResult`
- [ ] Create `sivo/providers/registry.py` with `get_provider()` resolution
- [ ] Create `sivo/providers/anthropic.py` — extract existing code from execution engine and judge into `AnthropicProvider`
- [ ] Update `SivoConfig` with provider fields
- [ ] Update execution engine and judge to use provider via `complete()` / `judge()`
- [ ] Ensure all existing tests still pass (this is a refactor — zero behaviour change)

### Phase B — OpenAI provider (1 session)

- [ ] Create `sivo/providers/openai.py` — implement `OpenAIProvider`
- [ ] Implement function-calling judge for OpenAI
- [ ] Implement prompt-based JSON fallback judge utility
- [ ] Add `openai` as an optional dependency (`pip install sivo[openai]`)
- [ ] Write unit tests with mocked OpenAI client
- [ ] Write integration test: real OpenAI call (guarded by `--no-llm` + env var check)

### Phase C — CLI + config + e2e (1 session)

- [ ] Wire `--provider` and `--judge-provider` CLI flags
- [ ] Wire `sivo.toml` provider config loading
- [ ] Update cache key to include model
- [ ] E2e test: run same eval with both providers (mocked) and verify identical pipeline behaviour
- [ ] E2e test: custom provider via import path
- [ ] Update README with multi-provider docs
- [ ] Update DECISIONS.md — mark Decision 14 as resolved

---

## Dependencies

- `anthropic` — already a dependency (stays required)
- `openai` — new optional dependency via extras: `pip install sivo[openai]`
- No other new dependencies

---

## Resolved decisions

1. **Cost tables** — DECIDED: Hardcode per-model pricing with a config override. No external pricing API dependency.
2. **Cache key** — DECIDED: Include model in the cache key to prevent false hits across providers.
3. **Streaming** — DECIDED: No streaming for v1. Eval calls are batch-oriented; defer to a future version.
4. **Provider-specific features** — DECIDED: Ignore for now. Providers expose these via `raw_response` for advanced users.
