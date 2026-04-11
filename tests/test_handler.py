"""Tests for handler.py — no GPU, Triton process, or model weights required.

All external I/O is mocked:
  - Triton HTTP calls  → httpx patched
  - tokenizer         → handler._tokenizer set directly
  - subprocess        → patched so tritonserver never spawns
  - generate_config   → patched so no filesystem writes
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Module fixture ────────────────────────────────────────────────────────────

@pytest.fixture()
def h(monkeypatch):
    """Import handler with a clean environment and no side-effecting imports."""
    monkeypatch.setenv("MODEL_PATH", "/vol/models/gemma")
    monkeypatch.setenv("MODEL_NAME", "model")
    monkeypatch.setenv("TRITON_HTTP_PORT", "8000")
    monkeypatch.setenv("TRITON_STARTUP_TIMEOUT", "1")

    sys.modules.pop("handler", None)
    sys.modules.pop("generate_config", None)

    mod = importlib.import_module("handler")
    mod._tokenizer = None
    mod._triton_proc = None
    yield mod
    mod._tokenizer = None
    mod._triton_proc = None
    sys.modules.pop("handler", None)
    sys.modules.pop("generate_config", None)


@pytest.fixture()
def tokenizer(h):
    """Attach a mock tokenizer to the handler module."""
    tok = MagicMock()
    tok.apply_chat_template.return_value = "<bos><start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n"
    h._tokenizer = tok
    return tok


# ── _sampling_params ──────────────────────────────────────────────────────────

class TestSamplingParams:
    def test_all_defaults(self, h):
        p = h._sampling_params({})
        assert p["max_tokens"] == 512
        assert p["temperature"] == 0.7
        assert p["top_p"] == 1.0
        assert p["top_k"] == -1
        assert p["stop"] == []
        assert p["frequency_penalty"] == 0.0
        assert p["presence_penalty"] == 0.0

    def test_partial_override(self, h):
        p = h._sampling_params({"max_tokens": 1024, "temperature": 0.0})
        assert p["max_tokens"] == 1024
        assert p["temperature"] == 0.0
        assert p["top_p"] == 1.0  # default unchanged

    def test_stop_sequences(self, h):
        p = h._sampling_params({"stop": ["</s>", "\n\n"]})
        assert p["stop"] == ["</s>", "\n\n"]

    def test_all_fields_overridable(self, h):
        overrides = {
            "max_tokens": 2048,
            "temperature": 1.2,
            "top_p": 0.95,
            "top_k": 50,
            "stop": ["<eos>"],
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


# ── _apply_chat_template ──────────────────────────────────────────────────────

class TestApplyChatTemplate:
    def test_calls_apply_chat_template(self, h, tokenizer):
        messages = [{"role": "user", "content": "hi"}]
        h._apply_chat_template(messages)
        tokenizer.apply_chat_template.assert_called_once_with(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def test_returns_formatted_string(self, h, tokenizer):
        result = h._apply_chat_template([{"role": "user", "content": "hi"}])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_passes_messages_unchanged(self, h, tokenizer):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
        h._apply_chat_template(messages)
        actual_messages = tokenizer.apply_chat_template.call_args[0][0]
        assert actual_messages == messages


# ── handler — non-streaming ───────────────────────────────────────────────────

class TestHandlerNonStreaming:
    async def test_missing_messages_returns_error(self, h, tokenizer):
        result = await h.handler({"input": {}})
        assert "error" in result

    async def test_basic_response_shape(self, h, tokenizer):
        with patch("handler._query", new=AsyncMock(return_value="Hi there!")):
            result = await h.handler({"input": {"messages": [{"role": "user", "content": "hi"}]}})
        assert result["choices"][0]["message"]["content"] == "Hi there!"
        assert result["choices"][0]["finish_reason"] == "stop"

    async def test_sampling_params_forwarded(self, h, tokenizer):
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

    async def test_prompt_from_chat_template(self, h, tokenizer):
        with patch("handler._query", new=AsyncMock(return_value="ok")) as mock_q:
            await h.handler({"input": {"messages": [{"role": "user", "content": "hi"}]}})
        prompt, _ = mock_q.call_args[0]
        assert prompt == tokenizer.apply_chat_template.return_value

    async def test_empty_messages_list_returns_error(self, h, tokenizer):
        result = await h.handler({"input": {"messages": []}})
        assert "error" in result


# ── handler — streaming ───────────────────────────────────────────────────────

class TestHandlerStreaming:
    async def _collect(self, gen):
        chunks = []
        async for chunk in gen:
            chunks.append(chunk)
        return chunks

    async def test_returns_async_generator(self, h, tokenizer):
        async def fake_stream(prompt, params):
            yield "hello"

        with patch("handler._stream", new=fake_stream):
            result = await h.handler({
                "input": {
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
            })
        # Must be an async iterable
        import inspect
        assert inspect.isasyncgen(result)

    async def test_chunk_content(self, h, tokenizer):
        tokens = ["The", " sky", " is", " blue", "."]

        async def fake_stream(prompt, params):
            for t in tokens:
                yield t

        with patch("handler._stream", new=fake_stream):
            result = await h.handler({
                "input": {
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
            })
            # Iterate inside the patch block: generate() is lazy and looks up
            # _stream at iteration time, not at the point handler() is called.
            chunks = await self._collect(result)

        content_chunks = chunks[:-1]  # last is the finish sentinel
        assert [c["choices"][0]["delta"]["content"] for c in content_chunks] == tokens

    async def test_final_chunk_has_finish_reason_stop(self, h, tokenizer):
        async def fake_stream(prompt, params):
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

    async def test_intermediate_chunks_have_no_finish_reason(self, h, tokenizer):
        async def fake_stream(prompt, params):
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

    async def test_sampling_params_forwarded(self, h, tokenizer):
        received = {}

        async def fake_stream(prompt, params):
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
    def test_generates_config_before_starting_triton(self, h):
        call_order = []

        def fake_generate():
            call_order.append("generate")
            return "/model_repo", "model"

        def fake_start(repo):
            call_order.append("start")

        with (
            patch("handler.generate_model_repo", side_effect=fake_generate),
            patch("handler._start_triton", side_effect=fake_start),
            patch("handler._load_tokenizer"),
            patch("handler._wait_for_triton"),
        ):
            h.init()

        assert call_order == ["generate", "start"]

    def test_waits_for_triton_after_start(self, h):
        call_order = []

        with (
            patch("handler.generate_model_repo", return_value=("/repo", "model")),
            patch("handler._start_triton"),
            patch("handler._load_tokenizer", side_effect=lambda: call_order.append("tokenizer")),
            patch("handler._wait_for_triton", side_effect=lambda: call_order.append("wait")),
        ):
            h.init()

        # tokenizer loads concurrently with triton (before wait)
        assert call_order.index("tokenizer") < call_order.index("wait")

    def test_triton_timeout_raises(self, h, monkeypatch):
        import time

        monkeypatch.setenv("TRITON_STARTUP_TIMEOUT", "0")
        sys.modules.pop("handler", None)
        mod = importlib.import_module("handler")
        mod._tokenizer = MagicMock()

        with (
            patch("handler.generate_model_repo", return_value=("/repo", "model")),
            patch("handler._start_triton"),
            patch("handler._load_tokenizer"),
            patch("httpx.get", side_effect=ConnectionRefusedError),
        ):
            with pytest.raises(RuntimeError, match="ready"):
                mod._wait_for_triton()
