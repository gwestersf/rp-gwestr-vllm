"""RunPod serverless handler — Triton + vLLM, OpenAI-compatible chat interface.

Input schema:
    {
        "messages": [{"role": "user", "content": "..."}],  # required
        "max_tokens": 512,       # optional
        "temperature": 0.7,      # optional
        "top_p": 1.0,            # optional
        "top_k": -1,             # optional
        "stop": [],              # optional
        "frequency_penalty": 0.0,# optional
        "presence_penalty": 0.0, # optional
        "stream": false          # optional — set true for SSE streaming
    }
"""

import json
import os
import subprocess
import time

import httpx
import runpod

from generate_config import generate_model_repo

# ── Config ────────────────────────────────────────────────────────────────────

TRITON_HTTP_PORT = int(os.environ.get("TRITON_HTTP_PORT", "8000"))
TRITON_BASE = f"http://localhost:{TRITON_HTTP_PORT}"
MODEL_NAME = os.environ.get("MODEL_NAME", "model")
STARTUP_TIMEOUT = int(os.environ.get("TRITON_STARTUP_TIMEOUT", "1800"))

_tokenizer = None
_triton_proc = None


# ── Startup ───────────────────────────────────────────────────────────────────

def _start_triton(model_repo: str) -> None:
    global _triton_proc
    cmd = [
        "tritonserver",
        f"--model-repository={model_repo}",
        f"--http-port={TRITON_HTTP_PORT}",
        "--grpc-port=8001",
        "--metrics-port=8002",
        "--log-verbose=0",
    ]
    print(f"[triton] Starting: {' '.join(cmd)}")
    _triton_proc = subprocess.Popen(cmd)


def _wait_for_triton() -> None:
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        try:
            r = httpx.get(f"{TRITON_BASE}/v2/health/ready", timeout=5)
            if r.status_code == 200:
                print("[triton] Ready")
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"Triton did not become ready within {STARTUP_TIMEOUT}s")


def _load_tokenizer() -> None:
    global _tokenizer
    model_path = os.environ["MODEL_PATH"]
    from transformers import AutoTokenizer
    print(f"[tokenizer] Loading from {model_path}")
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("[tokenizer] Loaded")


def init() -> None:
    model_repo, _ = generate_model_repo()
    _start_triton(model_repo)
    # Load tokenizer concurrently while Triton warms up.
    _load_tokenizer()
    _wait_for_triton()
    print("[init] Ready to serve")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_chat_template(messages: list) -> str:
    return _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _sampling_params(job_input: dict) -> dict:
    return {
        "max_tokens": job_input.get("max_tokens", 512),
        "temperature": job_input.get("temperature", 0.7),
        "top_p": job_input.get("top_p", 1.0),
        "top_k": job_input.get("top_k", -1),
        "stop": job_input.get("stop", []),
        "frequency_penalty": job_input.get("frequency_penalty", 0.0),
        "presence_penalty": job_input.get("presence_penalty", 0.0),
    }


def _choice(content: str, finish_reason=None) -> dict:
    return {
        "choices": [{
            "delta": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }]
    }


# ── Triton calls ──────────────────────────────────────────────────────────────

async def _query(prompt: str, params: dict) -> str:
    payload = {"text_input": prompt, "parameters": {**params, "stream": False}}
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(
            f"{TRITON_BASE}/v2/models/{MODEL_NAME}/generate", json=payload
        )
        r.raise_for_status()
        return r.json()["text_output"]


async def _stream(prompt: str, params: dict):
    """Yields incremental text deltas from Triton's SSE stream."""
    payload = {"text_input": prompt, "parameters": {**params, "stream": True}}
    prev_len = 0
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST",
            f"{TRITON_BASE}/v2/models/{MODEL_NAME}/generate_stream",
            json=payload,
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                full_text = json.loads(data_str).get("text_output", "")
                delta = full_text[prev_len:]
                prev_len = len(full_text)
                if delta:
                    yield delta


# ── RunPod handler ────────────────────────────────────────────────────────────

async def handler(job):
    job_input = job["input"]
    messages = job_input.get("messages")
    if not messages:
        return {"error": "'messages' is required"}

    prompt = _apply_chat_template(messages)
    params = _sampling_params(job_input)

    if job_input.get("stream", False):
        async def generate():
            async for token in _stream(prompt, params):
                yield _choice(token)
            yield _choice("", finish_reason="stop")
        return generate()

    text = await _query(prompt, params)
    return {
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }]
    }


if __name__ == "__main__":
    init()
    runpod.serverless.start({"handler": handler})
