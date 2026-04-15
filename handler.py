"""RunPod serverless handler — vLLM in-process via Python API.

vLLM is initialized directly in the worker process — no subprocess, no HTTP
roundtrip. Engine args are fully configurable from environment variables;
see engine_args.py for the full list.

Input schema
------------
{
    "messages": [{"role": "user", "content": "..."}],   # required
    "max_tokens":         512,
    "temperature":        0.7,
    "top_p":              1.0,
    "top_k":              -1,
    "stop":               [],
    "frequency_penalty":  0.0,
    "presence_penalty":   0.0,
    "stream":             false
}

Test mode
---------
{"_run_tests": true}
"""

import os
import sys
import traceback
import uuid

import runpod

MODEL_PATH = os.environ["MODEL_PATH"]

_engine = None
_tokenizer = None


# ── Startup ───────────────────────────────────────────────────────────────────

def init() -> None:
    global _engine, _tokenizer
    from vllm import AsyncLLMEngine
    from transformers import AutoTokenizer
    from engine_args import get_engine_args

    print("[init] Loading engine...")
    _engine = AsyncLLMEngine.from_engine_args(get_engine_args())
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("[init] Ready to serve")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sampling_params(job_input: dict) -> dict:
    params = {
        "max_tokens": job_input.get("max_tokens", 512),
        "temperature": job_input.get("temperature", 0.7),
        "top_p": job_input.get("top_p", 1.0),
        "frequency_penalty": job_input.get("frequency_penalty", 0.0),
        "presence_penalty": job_input.get("presence_penalty", 0.0),
    }
    if job_input.get("stop"):
        params["stop"] = job_input["stop"]
    if job_input.get("top_k", -1) != -1:
        params["top_k"] = job_input["top_k"]
    return params


def _choice(content: str, finish_reason=None) -> dict:
    return {
        "choices": [{
            "delta": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }]
    }


# ── Inference ─────────────────────────────────────────────────────────────────

async def _query(messages: list, params: dict) -> str:
    from vllm import SamplingParams
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    final_output = None
    async for output in _engine.generate(prompt, SamplingParams(**params), str(uuid.uuid4())):
        final_output = output
    return final_output.outputs[0].text


async def _stream(messages: list, params: dict):
    from vllm import SamplingParams
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prev_len = 0
    async for output in _engine.generate(prompt, SamplingParams(**params), str(uuid.uuid4())):
        text = output.outputs[0].text
        delta = text[prev_len:]
        prev_len = len(text)
        if delta:
            yield delta


# ── Endpoint tests ────────────────────────────────────────────────────────────

async def _run_tests(job_input: dict) -> dict:
    results = {}

    # Health
    results["health"] = "ok" if _engine is not None else "failed: engine not initialized"

    # Chat sanity check
    try:
        messages = [{
            "role": "user",
            "content": (
                "Explain how the CUDA memory hierarchy works — specifically how data "
                "moves between global memory, L2 cache, L1/shared memory, and registers "
                "during a typical matrix multiplication kernel. Include how warp-level "
                "memory access patterns affect coalescing and why misaligned access "
                "hurts throughput."
            ),
        }]
        text = await _query(messages, {
            "max_tokens": 512, "temperature": 0.0,
            "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0,
        })
        results["chat"] = f"ok — response: {text[:200]!r}..."
    except Exception as e:
        results["chat"] = f"failed: {e}"

    return results


# ── RunPod handler ────────────────────────────────────────────────────────────

async def _handle(job):
    job_input = job["input"]

    if job_input.get("_run_tests"):
        return await _run_tests(job_input)

    messages = job_input.get("messages")
    if not messages:
        return {"error": "'messages' is required"}

    params = _sampling_params(job_input)

    if job_input.get("stream", False):
        async def generate():
            async for token in _stream(messages, params):
                yield _choice(token)
            yield _choice("", finish_reason="stop")
        return generate()

    text = await _query(messages, params)
    return {
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }]
    }


def handler(event):
    return _handle(event)


if __name__ == "__main__":
    try:
        init()
    except Exception as e:
        print(f"[init] Failed: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
    runpod.serverless.start({"handler": handler})
