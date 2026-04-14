"""RunPod serverless handler — vLLM OpenAI-compatible server.

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

import os
import subprocess
import time

import httpx
import runpod

# ── Config ────────────────────────────────────────────────────────────────────

VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
VLLM_BASE = f"http://localhost:{VLLM_PORT}"
MODEL_PATH = os.environ["MODEL_PATH"]
STARTUP_TIMEOUT = int(os.environ.get("TRITON_STARTUP_TIMEOUT", "1800"))

_vllm_proc = None


# ── Startup ───────────────────────────────────────────────────────────────────

def _start_vllm() -> None:
    global _vllm_proc
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", os.environ.get("TENSOR_PARALLEL_SIZE", "1"),
        "--max-model-len", os.environ.get("MAX_MODEL_LEN", "8192"),
        "--gpu-memory-utilization", os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"),
        "--max-num-seqs", os.environ.get("MAX_NUM_SEQS", "4"),
        "--trust-remote-code",
    ]
    print(f"[vllm] Starting: {' '.join(cmd)}")
    _vllm_proc = subprocess.Popen(cmd)


def _wait_for_vllm() -> None:
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        try:
            r = httpx.get(f"{VLLM_BASE}/health", timeout=5)
            if r.status_code == 200:
                print("[vllm] Ready")
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"vLLM did not become ready within {STARTUP_TIMEOUT}s")


def init() -> None:
    _start_vllm()
    _wait_for_vllm()
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


# ── vLLM OpenAI calls ─────────────────────────────────────────────────────────

async def _query(messages: list, params: dict) -> str:
    payload = {"model": MODEL_PATH, "messages": messages, "stream": False, **params}
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{VLLM_BASE}/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def _stream(messages: list, params: dict):
    """Yields incremental text deltas from vLLM's SSE stream."""
    payload = {"model": MODEL_PATH, "messages": messages, "stream": True, **params}
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST", f"{VLLM_BASE}/v1/chat/completions", json=payload
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                import json
                delta = json.loads(data_str)["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta


# ── RunPod handler ────────────────────────────────────────────────────────────

async def _handle(job):
    job_input = job["input"]
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


async def handler(event):
    return await _handle(event)


if __name__ == "__main__":
    init()
    runpod.serverless.start({"handler": handler})
