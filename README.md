# NVIDIA Dynamo with vLLM OpenAI Server

[![Runpod](https://api.runpod.io/badge/gwestersf/rp-gwestr-vllm)](https://console.runpod.io/hub/gwestersf/rp-gwestr-vllm)

RunPod serverless endpoint that serves any Hugging Face model with an OpenAI-compatible chat API, powered by [vLLM](https://github.com/vllm-project/vllm) 0.19.0.

Scale from small to large models by adjusting environment variables — no code changes, no image rebuild required.

---

## Requirements

- NVIDIA Blackwell GPU (RTX PRO 6000, B200, etc.)
- NVIDIA driver **575+** (CUDA 13.0)
- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

## How it works

On worker startup:
1. vLLM's OpenAI-compatible API server launches, loading the model from `MODEL_PATH`
2. The handler polls `/health` until the server is ready
3. Incoming RunPod jobs are forwarded to `/v1/chat/completions`

---

## Inference backends

This project is built on [NVIDIA AI Dynamo](https://github.com/ai-dynamo/dynamo), which natively supports both **vLLM** and **TensorRT-LLM** as inference backends. The current image uses vLLM for broad model compatibility and ease of use. Swapping to TensorRT-LLM is an architectural option within the same Dynamo runtime — no change to the RunPod handler or serving infrastructure required, only the backend component and model format.

| Backend | Strengths |
|---|---|
| **vLLM** (current) | Any Hugging Face model, fast iteration, FP16/BF16, continuous batching |
| **TensorRT-LLM** | Maximum throughput, INT8/FP8 quantization, optimized CUDA kernels for production |

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | **required** | Path to model weights (e.g. `/runpod-volume/models/gemma-4-E4B-it`) |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs to shard across — set to `2` for 26B, `4`+ for 70B+ |
| `MAX_MODEL_LEN` | `8192` | Max context window in tokens. Reduce if you get OOM on startup. |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of VRAM reserved for KV cache |
| `MAX_NUM_SEQS` | `4` | Max concurrent requests per worker |
| `VLLM_PORT` | `8000` | Internal port vLLM listens on |
| `VLLM_STARTUP_TIMEOUT` | `1800` | Seconds to wait for vLLM to become ready — increase for large models |

### GPU sizing guide

| Model size | VRAM (BF16) | `TENSOR_PARALLEL_SIZE` | Example GPU |
|---|---|---|---|
| 4B | ~10 GB | 1 | RTX PRO 6000 Blackwell (48 GB) |
| 8B | ~16 GB | 1 | RTX PRO 6000 Blackwell (48 GB) |
| 26B | ~52 GB | 2 | 2× RTX PRO 6000 Blackwell |
| 70B | ~140 GB | 4 | 4× A100 80 GB |
| 100B+ MoE | varies | 4–8 | 4–8× H100/A100 |

---

## RunPod deployment

1. Create a serverless endpoint using the image `gwesterrunpod/rp-gwestr-vllm:<version>`
2. Attach a network volume (50 GB+) mounted at `/runpod-volume`
3. Set `MODEL_PATH` to the location of your model weights on the volume
4. Pre-load model weights onto the network volume before starting the endpoint

---

## Running locally

### Single GPU

```bash
docker run --rm --gpus all \
  -v /path/to/model:/model:ro \
  -p 8000:8000 \
  -e MODEL_PATH=/model \
  -e MAX_MODEL_LEN=8192 \
  -e GPU_MEMORY_UTILIZATION=0.92 \
  -e MAX_NUM_SEQS=8 \
  gwesterrunpod/rp-gwestr-vllm:0.2.9
```

### Multi-GPU (26B+ models)

```bash
docker run --rm --gpus all \
  -v /path/to/model:/model:ro \
  -p 8000:8000 \
  -e MODEL_PATH=/model \
  -e TENSOR_PARALLEL_SIZE=2 \
  -e MAX_MODEL_LEN=8192 \
  -e GPU_MEMORY_UTILIZATION=0.90 \
  -e MAX_NUM_SEQS=4 \
  gwesterrunpod/rp-gwestr-vllm:0.2.9
```

### Test prompt

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Building

Bump `VERSION`, then run:

```bash
./release.sh
```

This builds and pushes `gwesterrunpod/rp-gwestr-vllm:<version>`. Published images are available at [hub.docker.com/repository/docker/gwesterrunpod/rp-gwestr-vllm](https://hub.docker.com/repository/docker/gwesterrunpod/rp-gwestr-vllm/general). Pin the version tag in your RunPod endpoint template — avoid `latest` so deployments are reproducible and rollback is straightforward.

---

## Endpoint tests

Send `"_run_tests": true` as the input to run a health check, chat sanity check, and aiperf performance benchmark directly inside the worker — no external tooling required.

```json
{
  "input": {
    "_run_tests": true,
    "concurrency": 2,
    "requests": 10
  }
}
```

`concurrency` and `requests` are optional (defaults: 2 and 10). With curl:

```bash
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{"input": {"_run_tests": true, "concurrency": 2, "requests": 10}}'
```

The response includes:
- `health` — vLLM `/health` status
- `chat` — single completion sanity check and response
- `aiperf` — full aiperf metrics table (TTFT, ITL, throughput, etc.)

---

## Local testing

### Unit tests (no GPU required)

```bash
pip install pytest pytest-asyncio
pytest tests/test_handler.py -v
```

### Integration tests (GPU + live server required)

Requires a running vLLM server and [`aiperf`](https://github.com/ai-dynamo/aiperf) (`uv tool install aiperf --python 3.12`).

```bash
MODEL_PATH=/path/to/model \
VLLM_URL=http://localhost:8000 \
pytest tests/test_integration.py -v -s
```

| Variable | Default | Description |
|---|---|---|
| `VLLM_URL` | `http://localhost:8000` | vLLM server URL |
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
