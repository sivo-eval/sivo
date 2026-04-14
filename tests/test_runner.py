"""Unit tests for sivo.runner (ExecutionEngine).

All tests use a mocked Anthropic client — no real LLM calls are made.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sivo.models import ExecutionInput, ExecutionRecord, Message
from sivo.runner import ExecutionEngine, _build_messages, _calculate_cost, _generate_run_id
from sivo.store import JsonlStore


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _mock_response(output: str = "Test output", input_tokens: int = 10, output_tokens: int = 5):
    """Return a mock Anthropic Messages response."""
    response = MagicMock()
    response.content = [MagicMock(text=output)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return response


def _make_engine(tmp_path, **kwargs) -> ExecutionEngine:
    store = JsonlStore(tmp_path / ".sivo")
    return ExecutionEngine(store=store, api_key="test-key", **kwargs)


def _make_input(**overrides) -> ExecutionInput:
    defaults = dict(input="What is 2+2?")
    defaults.update(overrides)
    return ExecutionInput(**defaults)


# ---------------------------------------------------------------------------
# _calculate_cost
# ---------------------------------------------------------------------------


def test_cost_known_model():
    cost = _calculate_cost("claude-haiku-4-5", 1_000_000, 1_000_000)
    assert cost == pytest.approx(4.80)  # 0.80 + 4.00


def test_cost_zero_tokens():
    assert _calculate_cost("claude-haiku-4-5", 0, 0) == 0.0


def test_cost_unknown_model_uses_fallback():
    # Unknown model uses Sonnet pricing (3.00 + 15.00) / 1M
    cost = _calculate_cost("unknown-model-xyz", 1_000_000, 1_000_000)
    assert cost == pytest.approx(18.00)


# ---------------------------------------------------------------------------
# _generate_run_id
# ---------------------------------------------------------------------------


def test_run_id_format():
    run_id = _generate_run_id()
    assert run_id.startswith("run_")
    # Should have the form run_YYYYMMDD_HHMMSS_<hex>
    parts = run_id.split("_")
    assert len(parts) == 4


def test_run_ids_are_unique():
    ids = {_generate_run_id() for _ in range(100)}
    assert len(ids) == 100


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------


def test_build_messages_single_turn_str():
    spec = ExecutionInput(input="Hello there")
    msgs = _build_messages(spec)
    assert msgs == [{"role": "user", "content": "Hello there"}]


def test_build_messages_single_turn_dict():
    spec = ExecutionInput(input={"question": "What?"})
    msgs = _build_messages(spec)
    assert msgs == [{"role": "user", "content": "{'question': 'What?'}"}]


def test_build_messages_conversation():
    spec = ExecutionInput(
        input="ignored when conversation is set",
        conversation=[
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
            Message(role="user", content="How are you?"),
        ],
    )
    msgs = _build_messages(spec)
    assert len(msgs) == 3
    assert msgs[0] == {"role": "user", "content": "Hi"}
    assert msgs[1] == {"role": "assistant", "content": "Hello!"}
    assert msgs[2] == {"role": "user", "content": "How are you?"}


# ---------------------------------------------------------------------------
# ExecutionEngine.execute — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_returns_execution_record(tmp_path):
    mock_resp = _mock_response("Four.")
    engine = _make_engine(tmp_path)

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        spec = _make_input(input="What is 2+2?")
        record = await engine.execute(spec, run_id="run_test")

    assert isinstance(record, ExecutionRecord)
    assert record.output == "Four."
    assert record.run_id == "run_test"
    assert record.input == "What is 2+2?"
    assert record.model == engine.model
    assert record.input_tokens == 10
    assert record.output_tokens == 5


@pytest.mark.asyncio
async def test_execute_writes_record_to_store(tmp_path):
    mock_resp = _mock_response("Hello")
    engine = _make_engine(tmp_path)

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        spec = _make_input()
        await engine.execute(spec, run_id="run_written")

    records = engine.store.read("run_written")
    assert len(records) == 1
    assert records[0].output == "Hello"


@pytest.mark.asyncio
async def test_execute_includes_system_prompt(tmp_path):
    mock_resp = _mock_response()
    engine = _make_engine(tmp_path)

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        spec = _make_input(system_prompt="You are a calculator.")
        record = await engine.execute(spec, run_id="run_sys")

    assert record.system_prompt == "You are a calculator."

    # Verify system prompt was passed to the API call
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs.get("system") == "You are a calculator."


@pytest.mark.asyncio
async def test_execute_passes_extra_params(tmp_path):
    mock_resp = _mock_response()
    engine = _make_engine(tmp_path)

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        spec = _make_input(params={"temperature": 0.5, "max_tokens": 256})
        await engine.execute(spec, run_id="run_params")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs.get("temperature") == 0.5
    assert call_kwargs.get("max_tokens") == 256


@pytest.mark.asyncio
async def test_execute_records_cost(tmp_path):
    mock_resp = _mock_response(input_tokens=100, output_tokens=50)
    engine = _make_engine(tmp_path, model="claude-haiku-4-5")

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        record = await engine.execute(_make_input(), run_id="run_cost")

    expected = (100 * 0.80 + 50 * 4.00) / 1_000_000
    assert record.cost_usd == pytest.approx(expected)


@pytest.mark.asyncio
async def test_execute_metadata_preserved(tmp_path):
    mock_resp = _mock_response()
    engine = _make_engine(tmp_path)

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        spec = _make_input(metadata={"tag": "smoke", "dataset": "v1"})
        record = await engine.execute(spec, run_id="run_meta")

    assert record.metadata == {"tag": "smoke", "dataset": "v1"}


# ---------------------------------------------------------------------------
# run_many — concurrency and auto run_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_many_returns_all_records(tmp_path):
    responses = [_mock_response(f"Out {i}") for i in range(5)]
    engine = _make_engine(tmp_path)

    call_count = 0

    async def fake_create(**kwargs):
        nonlocal call_count
        r = responses[call_count]
        call_count += 1
        return r

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = fake_create

        specs = [_make_input(input=f"Q{i}") for i in range(5)]
        records = await engine.run_many(specs, run_id="run_many")

    assert len(records) == 5
    assert all(isinstance(r, ExecutionRecord) for r in records)
    assert all(r.run_id == "run_many" for r in records)


@pytest.mark.asyncio
async def test_run_many_auto_generates_run_id(tmp_path):
    mock_resp = _mock_response()
    engine = _make_engine(tmp_path)

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        records = await engine.run_many([_make_input()], run_id=None)

    assert records[0].run_id.startswith("run_")


@pytest.mark.asyncio
async def test_run_many_writes_all_to_store(tmp_path):
    mock_resp = _mock_response()
    engine = _make_engine(tmp_path)

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        specs = [_make_input(input=f"Q{i}") for i in range(3)]
        await engine.run_many(specs, run_id="run_store")

    assert len(engine.store.read("run_store")) == 3


@pytest.mark.asyncio
async def test_concurrency_limit_respected(tmp_path):
    """Verify the semaphore never allows more than N simultaneous calls."""
    max_concurrent = 0
    current = 0

    async def fake_create(**kwargs):
        nonlocal max_concurrent, current
        current += 1
        max_concurrent = max(max_concurrent, current)
        await asyncio.sleep(0.01)  # simulate I/O
        current -= 1
        return _mock_response()

    engine = _make_engine(tmp_path, concurrency=3)

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = fake_create

        specs = [_make_input() for _ in range(10)]
        await engine.run_many(specs, run_id="run_conc")

    assert max_concurrent <= 3


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retries_on_transient_error(tmp_path):
    """A transient failure on the first attempt should succeed on the second."""
    mock_resp = _mock_response()
    attempt = 0

    async def fake_create(**kwargs):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise ConnectionError("transient error")
        return mock_resp

    engine = _make_engine(tmp_path, retries=3)

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = fake_create

        with patch("asyncio.sleep", new_callable=AsyncMock):
            record = await engine.execute(_make_input(), run_id="run_retry")

    assert record.output == "Test output"
    assert attempt == 2


@pytest.mark.asyncio
async def test_raises_after_all_retries_exhausted(tmp_path):
    engine = _make_engine(tmp_path, retries=3)

    async def always_fail(**kwargs):
        raise ValueError("permanent failure")

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = always_fail

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ValueError, match="permanent failure"):
                await engine.execute(_make_input(), run_id="run_fail")


# ---------------------------------------------------------------------------
# Timeout behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_raises_timeout_error(tmp_path):
    async def slow_call(**kwargs):
        await asyncio.sleep(10)
        return _mock_response()

    engine = _make_engine(tmp_path, timeout=0.05, retries=1)

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = slow_call

        with pytest.raises(TimeoutError):
            await engine.execute(_make_input(), run_id="run_timeout")


# ---------------------------------------------------------------------------
# Integration test (live LLM — skipped with --no-llm)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_real_llm_round_trip(tmp_path, request):
    if request.config.getoption("--no-llm"):
        pytest.skip("--no-llm: skipping live LLM integration test")

    store = JsonlStore(tmp_path / ".sivo")
    engine = ExecutionEngine(store=store)

    spec = ExecutionInput(input="Reply with exactly one word: hello")
    record = await engine.execute(spec, run_id="run_live")

    assert isinstance(record, ExecutionRecord)
    assert len(record.output) > 0
    assert record.input_tokens > 0
    assert record.output_tokens > 0
    assert record.cost_usd >= 0

    # Verify it was written to JSONL
    stored = store.read("run_live")
    assert len(stored) == 1
    assert stored[0].id == record.id
