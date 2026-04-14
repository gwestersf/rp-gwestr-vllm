# rp-gwestr-vllm

[![Runpod](https://api.runpod.io/badge/gwestersf/rp-gwestr-vllm)](https://console.runpod.io/hub/gwestersf/rp-gwestr-vllm)

RunPod serverless endpoint that serves any Hugging Face model with an OpenAI-compatible chat API, powered by [vLLM](https://github.com/vllm-project/vllm).

Designed to scale from 4B to 100B+ models by adjusting env vars — no code changes, no image rebuild.

---

## Base image

Built from [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) `main`, which ships **vLLM 0.19.0**. The official `nvcr.io/nvidia/ai-dynamo/vllm-runtime` release tags ship an older vLLM version that doesn't support Gemma 4. We build from `main` until the next official release catches up.

The base image is published as `gwesterrunpod/dynamo-vllm-runtime:main` and requires:

- NVIDIA driver **575+** (CUDA 13.0)
- CUDA **13.0, 13.1, or 13.2**

> No Triton inference server. The handler starts vLLM's built-in OpenAI-compatible API server directly.

---

## How it works

On worker startup, `handler.py`:
1. Launches `python3 -m vllm.entrypoints.openai.api_server` pointed at `MODEL_PATH`
2. Polls `/health` until ready (up to `VLLM_STARTUP_TIMEOUT` seconds)
3. Forwards RunPod jobs to `/v1/chat/completions`

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | **required** | Path to model weights (e.g. `/runpod-volume/models/gemma-4-E4B-it`) |
| `TENSOR_PARALLEL_SIZE` | `1` | GPUs per worker — set to `2` for 26B, `4`+ for 70B+ |
| `MAX_MODEL_LEN` | `8192` | Max context window in tokens. Reduce if OOM on startup. |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of VRAM vLLM may use for KV cache |
| `MAX_NUM_SEQS` | `4` | Max concurrent requests per worker |
| `VLLM_PORT` | `8000` | Internal port (no need to change) |
| `VLLM_STARTUP_TIMEOUT` | `1800` | Seconds to wait for vLLM to become ready — increase for large models |

### Sizing guide

| Model size | VRAM (BF16) | `TENSOR_PARALLEL_SIZE` | GPU |
|---|---|---|---|
| 4B (e.g. Gemma 4 E4B) | ~10 GB | 1 | RTX PRO 6000 Blackwell (48 GB) |
| 8B | ~16 GB | 1 | RTX PRO 6000 Blackwell (48 GB) |
| 26B | ~52 GB | 2 | 2× RTX PRO 6000 Blackwell |
| 70B | ~140 GB | 4 | 4× A100 80 GB |
| 100B+ MoE | varies | 4–8 | 4–8× H100/A100 |

---

## Running locally

### Requirements

- NVIDIA driver **575+**
- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### RTX PRO 6000 Blackwell (48 GB VRAM)

```bash
docker run --rm --gpus all \
  -v /path/to/model:/model:ro \
  -p 8000:8000 \
  -e MODEL_PATH=/model \
  -e MAX_MODEL_LEN=8192 \
  -e GPU_MEMORY_UTILIZATION=0.92 \
  -e MAX_NUM_SEQS=8 \
  gwesterrunpod/rp-gwestr-vllm:0.2.8
```

For a 26B model across **2× RTX PRO 6000**:

```bash
docker run --rm --gpus all \
  -v /path/to/26b-model:/model:ro \
  -p 8000:8000 \
  -e MODEL_PATH=/model \
  -e TENSOR_PARALLEL_SIZE=2 \
  -e MAX_MODEL_LEN=8192 \
  -e GPU_MEMORY_UTILIZATION=0.90 \
  -e MAX_NUM_SEQS=4 \
  gwesterrunpod/rp-gwestr-vllm:0.2.8
```

### Test prompt

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model",
    "messages": [{"role": "user", "content": "How does RunPod serverless work?"}]
  }'
```

---

## Build and push

Bump `VERSION`, then run:

```bash
./release.sh
```

This builds and pushes `gwesterrunpod/rp-gwestr-vllm:<version>`. Never use `latest` — pin the version tag in your RunPod endpoint template so deployments are reproducible and rollback is a one-line change.

---

## RunPod deployment

1. Bump `VERSION` and run `./release.sh`
2. Create or update a serverless endpoint:
   - **Image:** `gwesterrunpod/rp-gwestr-vllm:<version>`
   - **Container disk:** 20 GB minimum
   - **Network volume:** mount at `/runpod-volume` (50 GB+)
3. Set env vars at the endpoint level — at minimum `MODEL_PATH`
4. Pre-load model weights onto the network volume before starting the endpoint

---

## Testing

### Unit tests (no GPU required)

Tests cover `handler.py` logic with all external I/O mocked — no vLLM process, no model weights needed.

```bash
pip install pytest pytest-asyncio
pytest tests/test_handler.py -v
```

### Integration tests (GPU + live server required)

Runs against a live vLLM server. Requires:
- NVIDIA GPU with `nvidia-smi`
- vLLM server running and healthy at `VLLM_URL`
- `MODEL_PATH` pointing to the local tokenizer (used by aiperf for token counting)
- [`aiperf`](https://github.com/ai-dynamo/aiperf) installed via `uv tool install aiperf --python 3.12`

```bash
MODEL_PATH=/path/to/model \
VLLM_URL=http://localhost:8000 \
pytest tests/test_integration.py -v -s
```

| Variable | Default | Description |
|---|---|---|
| `VLLM_URL` | `http://localhost:8000` | vLLM server base URL |
| `MODEL_PATH` | **required** | Local tokenizer path for aiperf token counting |
| `AIPERF_CONCURRENCY` | `2` | Parallel workers |
| `AIPERF_REQUESTS` | `10` | Total requests to send |

---

## Request format

```json
{
  "messages": [{"role": "user", "content": "Your prompt here"}],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 1.0,
  "top_k": -1,
  "stop": [],
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stream": false
}
```

`messages` is the only required field.

### Non-streaming response

```json
{
  "choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}]
}
```

### Streaming response (`"stream": true`)

```json
{"choices": [{"delta": {"role": "assistant", "content": "Hello"}, "finish_reason": null}]}
{"choices": [{"delta": {"role": "assistant", "content": " there"}, "finish_reason": null}]}
{"choices": [{"delta": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]}
```

---

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
