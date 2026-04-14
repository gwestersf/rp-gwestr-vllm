"""Integration tests — require a live vLLM server and NVIDIA GPU.

Run conditions:
  - NVIDIA GPU must be present (nvidia-smi check)
  - vLLM server must be reachable at VLLM_URL (default: http://localhost:8000)
  - MODEL_PATH must be set (local tokenizer path for aiperf)

Environment variables:
  VLLM_URL            vLLM server base URL   (default: http://localhost:8000)
  MODEL_PATH          Local tokenizer path   (required — for aiperf token counting)
  AIPERF_CONCURRENCY  Parallel workers       (default: 2)
  AIPERF_REQUESTS     Total requests to send (default: 10)
"""

import os
import shutil
import subprocess

import httpx
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and (
        subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=10
        ).returncode == 0
    )


def _server_url() -> str:
    return os.environ.get("VLLM_URL", "http://localhost:8000")


def _server_healthy() -> bool:
    try:
        r = httpx.get(f"{_server_url()}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _model_path() -> str:
    return os.environ.get("MODEL_PATH", "")


def _server_model_name() -> str:
    """Query the live server for the model name it has loaded."""
    try:
        r = httpx.get(f"{_server_url()}/v1/models", timeout=5)
        r.raise_for_status()
        return r.json()["data"][0]["id"]
    except Exception:
        return _model_path()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def require_gpu():
    if not _has_nvidia_gpu():
        pytest.fail("No NVIDIA GPU detected (nvidia-smi not found or failed). "
                    "Integration tests require a GPU machine.")


@pytest.fixture(scope="module", autouse=True)
def require_server(require_gpu):
    if not _server_healthy():
        pytest.fail(
            f"vLLM server not reachable at {_server_url()}/health. "
            "Start the server before running integration tests."
        )


@pytest.fixture(scope="module", autouse=True)
def require_model(require_gpu):
    if not _model_path():
        pytest.fail("MODEL_PATH env var is not set. "
                    "Set it to the model weights directory.")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_aiperf_profile(capsys):
    """Run aiperf against the live vLLM server and print results to log."""
    model = _server_model_name()   # model name as vLLM knows it
    tokenizer = _model_path()      # local path for aiperf token counting
    url = _server_url()
    concurrency = os.environ.get("AIPERF_CONCURRENCY", "2")
    requests = os.environ.get("AIPERF_REQUESTS", "10")

    cmd = [
        "aiperf", "profile",
        "--model", model,
        "--endpoint-type", "chat",
        "--url", url,
        "--tokenizer", tokenizer,
        "--streaming",
        "--concurrency", concurrency,
        "--request-count", requests,
    ]

    print(f"\n[aiperf] Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Always print output so it shows in pytest logs (-s or on failure)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    assert result.returncode == 0, (
        f"aiperf exited with code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_health_endpoint():
    """Sanity check: vLLM /health returns 200."""
    r = httpx.get(f"{_server_url()}/health", timeout=10)
    assert r.status_code == 200


def test_chat_completion():
    """Sanity check: single chat completion returns a non-empty response."""
    payload = {
        "model": _server_model_name(),
        "messages": [{"role": "user", "content": "Reply with one word: hello"}],
        "max_tokens": 16,
    }
    r = httpx.post(
        f"{_server_url()}/v1/chat/completions",
        json=payload,
        timeout=60,
    )
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert content.strip(), "Expected non-empty response from model"
    print(f"\n[chat] Response: {content.strip()!r}")
