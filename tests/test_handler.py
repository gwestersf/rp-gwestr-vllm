"""Tests for handler.py — no GPU, vLLM process, or model weights required.

All external dependencies are mocked:
  - vllm            → patched in sys.modules
  - transformers    → patched in sys.modules
  - engine_args     → patched in sys.modules
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Module fixture ────────────────────────────────────────────────────────────

@pytest.fixture()
def h(monkeypatch):
    """Import handler with a clean environment and all heavy deps mocked."""
    monkeypatch.setenv("MODEL_PATH", "/vol/models/gemma")

    # Mock vllm, transformers, and engine_args so no GPU or model is needed
    mock_vllm = MagicMock()
    mock_transformers = MagicMock()
    mock_engine_args_mod = MagicMock()
    mock_engine_args_mod.get_engine_args.return_value = MagicMock()

    for name in ("vllm", "transformers", "engine_args"):
        sys.modules.pop(name, None)

    monkeypatch.setitem(sys.modules, "vllm", mock_vllm)
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)
    monkeypatch.setitem(sys.modules, "engine_args", mock_engine_args_mod)

    sys.modules.pop("handler", None)
    mod = importlib.import_module("handler")

    # Pre-wire engine and tokenizer so tests don't need to call init()
    mod._engine = MagicMock()
    mod._tokenizer = MagicMock()
    mod._tokenizer.apply_chat_template.return_value = "<bos><user>hi</user><assistant>"

    yield mod

    mod._engine = None
    mod._tokenizer = None
    sys.modules.pop("handler", None)


# ── _sampling_params ──────────────────────────────────────────────────────────

class TestSamplingParams:
    def test_all_defaults(self, h):
        p = h._sampling_params({})
        assert p["max_tokens"] == 512
        assert p["temperature"] == 0.7
        assert p["top_p"] == 1.0
        assert p["frequency_penalty"] == 0.0
        assert p["presence_penalty"] == 0.0

    def test_top_k_excluded_when_minus_one(self, h):
        p = h._sampling_params({})
        assert "top_k" not in p

    def test_top_k_included_when_set(self, h):
        p = h._sampling_params({"top_k": 50})
        assert p["top_k"] == 50

    def test_stop_excluded_when_absent(self, h):
        p = h._sampling_params({})
        assert "stop" not in p

    def test_stop_included_when_set(self, h):
        p = h._sampling_params({"stop": ["</s>", "\n\n"]})
        assert p["stop"] == ["</s>", "\n\n"]

    def test_partial_override(self, h):
        p = h._sampling_params({"max_tokens": 1024, "temperature": 0.0})
        assert p["max_tokens"] == 1024
        assert p["temperature"] == 0.0
        assert p["top_p"] == 1.0

    def test_all_base_fields_overridable(self, h):
        overrides = {
            "max_tokens": 2048,
            "temperature": 1.2,
            "top_p": 0.95,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
        }
        p = h._sampling_params(overrides)
        for k, v in overrides.items():
            assert p[k] == v


# ── _choice ───────────────────────────────────────────────────────────────────

class TestChoice:
    def test_content_is_set(self, h):
        c = h._choice("hello world")
        assert c["choices"][0]["delta"]["content"] == "hello world"

    def test_role_is_assistant(self, h):
        c = h._choice("x")
        assert c["choices"][0]["delta"]["role"] == "assistant"

    def test_finish_reason_none_by_default(self, h):
        assert h._choice("x")["choices"][0]["finish_reason"] is None

    def test_finish_reason_stop(self, h):
        assert h._choice("", finish_reason="stop")["choices"][0]["finish_reason"] == "stop"

    def test_empty_content_allowed(self, h):
        c = h._choice("")
        assert c["choices"][0]["delta"]["content"] == ""


# ── handler — non-streaming ───────────────────────────────────────────────────

class TestHandlerNonStreaming:
    async def test_missing_messages_returns_error(self, h):
        result = await h.handler({"input": {}})
        assert "error" in result

    async def test_empty_messages_returns_error(self, h):
        result = await h.handler({"input": {"messages": []}})
        assert "error" in result

    async def test_none_messages_returns_error(self, h):
        result = await h.handler({"input": {"messages": None}})
        assert "error" in result

    async def test_basic_response_shape(self, h):
        with patch("handler._query", new=AsyncMock(return_value="Hi there!")):
            result = await h.handler({"input": {"messages": [{"role": "user", "content": "hi"}]}})
        assert result["choices"][0]["message"]["content"] == "Hi there!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["choices"][0]["message"]["role"] == "assistant"

    async def test_sampling_params_forwarded(self, h):
        with patch("handler._query", new=AsyncMock(return_value="ok")) as mock_q:
            await h.handler({
                "input": {
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 256,
                    "temperature": 0.1,
                    "top_p": 0.9,
                }
            })
        _, params = mock_q.call_args[0]
        assert params["max_tokens"] == 256
        assert params["temperature"] == 0.1
        assert params["top_p"] == 0.9

    async def test_messages_forwarded_to_query(self, h):
        messages = [{"role": "user", "content": "hello"}]
        with patch("handler._query", new=AsyncMock(return_value="ok")) as mock_q:
            await h.handler({"input": {"messages": messages}})
        forwarded_messages, _ = mock_q.call_args[0]
        assert forwarded_messages == messages


# ── handler — streaming ───────────────────────────────────────────────────────

class TestHandlerStreaming:
    async def _collect(self, gen):
        chunks = []
        async for chunk in gen:
            chunks.append(chunk)
        return chunks

    async def test_returns_async_generator(self, h):
        async def fake_stream(messages, params):
            yield "hello"

        with patch("handler._stream", new=fake_stream):
            result = await h.handler({
                "input": {
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
            })
        import inspect
        assert inspect.isasyncgen(result)

    async def test_chunk_content(self, h):
        tokens = ["The", " sky", " is", " blue", "."]

        async def fake_stream(messages, params):
            for t in tokens:
                yield t

        with patch("handler._stream", new=fake_stream):
            result = await h.handler({
                "input": {
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
            })
            chunks = await self._collect(result)

        content_chunks = chunks[:-1]
        assert [c["choices"][0]["delta"]["content"] for c in content_chunks] == tokens

    async def test_final_chunk_has_finish_reason_stop(self, h):
        async def fake_stream(messages, params):
            yield "hello"

        with patch("handler._stream", new=fake_stream):
            result = await h.handler({
                "input": {
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
            })
            chunks = await self._collect(result)

        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["choices"][0]["delta"]["content"] == ""

    async def test_intermediate_chunks_have_no_finish_reason(self, h):
        async def fake_stream(messages, params):
            for t in ["a", "b", "c"]:
                yield t

        with patch("handler._stream", new=fake_stream):
            result = await h.handler({
                "input": {
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
            })
            chunks = await self._collect(result)

        for chunk in chunks[:-1]:
            assert chunk["choices"][0]["finish_reason"] is None

    async def test_sampling_params_forwarded(self, h):
        received = {}

        async def fake_stream(messages, params):
            received.update(params)
            yield "ok"

        with patch("handler._stream", new=fake_stream):
            result = await h.handler({
                "input": {
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "max_tokens": 128,
                    "temperature": 0.5,
                }
            })
            await self._collect(result)

        assert received["max_tokens"] == 128
        assert received["temperature"] == 0.5


# ── init ──────────────────────────────────────────────────────────────────────

class TestInit:
    def test_engine_is_set(self, h):
        h._engine = None
        h._tokenizer = None
        h.init()
        assert h._engine is not None

    def test_tokenizer_is_set(self, h):
        h._engine = None
        h._tokenizer = None
        h.init()
        assert h._tokenizer is not None
