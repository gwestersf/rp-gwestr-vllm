# rp-gwestr-vllm

[![RunPod](https://api.runpod.io/badge/gwestersf/rp-gwestr-vllm)](https://console.runpod.io/hub/gwestersf/rp-gwestr-vllm)

OpenAI-compatible vLLM server for RunPod, built on [NVIDIA AI Dynamo](https://github.com/ai-dynamo/dynamo) (vLLM 0.19.0, CUDA 13.0). Serves any Hugging Face model with a drop-in OpenAI API — no code changes when swapping models, no image rebuild for config tuning.

**Docker image:** `gwesterrunpod/rp-gwestr-vllm:0.5.1`

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns 200 when ready, 204 while loading |
| `GET` | `/ping` | RunPod load-balancer health check (same as `/health`) |
| `GET` | `/v1/models` | List loaded model |
| `POST` | `/v1/chat/completions` | OpenAI chat completions |
| `POST` | `/v1/completions` | OpenAI text completions |
| `POST` | `/v1/responses` | OpenAI Responses API |
| `POST` | `/v1/messages` | Anthropic Messages API |

---

## Requirements

- NVIDIA GPU with 8 GB+ VRAM (Blackwell recommended for production)
- NVIDIA driver **575+** (CUDA 13.0)
- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

## Running locally

Single GPU with a Hugging Face model (downloaded on first run):

```bash
docker run --rm --gpus all -p 8080:80 \
  -e MODEL_PATH=google/gemma-3-1b-it \
  -e HF_TOKEN=hf_YOUR_TOKEN \
  -e MAX_MODEL_LEN=4096 \
  -e GPU_MEMORY_UTILIZATION=0.90 \
  -v ~/.cache/huggingface:/runpod-volume/huggingface-cache/hub \
  gwesterrunpod/rp-gwestr-vllm:0.5.1
```

With a locally downloaded model:

```bash
docker run --rm --gpus all -p 8080:80 \
  -e MODEL_PATH=/model \
  -e MAX_MODEL_LEN=8192 \
  -e GPU_MEMORY_UTILIZATION=0.92 \
  -e MAX_NUM_SEQS=8 \
  -v /path/to/model/weights:/model:ro \
  gwesterrunpod/rp-gwestr-vllm:0.5.1
```

Multi-GPU (e.g. 70B model across 4 GPUs):

```bash
docker run --rm --gpus all -p 8080:80 \
  -e MODEL_PATH=/model \
  -e TENSOR_PARALLEL_SIZE=4 \
  -e MAX_MODEL_LEN=8192 \
  -v /path/to/model/weights:/model:ro \
  gwesterrunpod/rp-gwestr-vllm:0.5.1
```

Once the server is ready (`/health` returns 200), send requests:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-1b-it",
    "messages": [{"role": "user", "content": "Explain what vLLM is in one paragraph."}],
    "max_tokens": 256
  }'
```

Streaming:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-1b-it",
    "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
    "max_tokens": 64,
    "stream": true
  }'
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | **required** | HuggingFace repo ID or absolute path to model weights |
| `HF_TOKEN` | — | HuggingFace token for gated models (e.g. Gemma, Llama) |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs to shard the model across |
| `MAX_MODEL_LEN` | `8192` | Max context length in tokens. Reduce to lower VRAM usage. |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of VRAM allocated for KV cache |
| `MAX_NUM_SEQS` | `4` | Max concurrent sequences per worker |
| `QUANTIZATION` | — | `fp8`, `bitsandbytes`, `awq`, etc. |
| `ENFORCE_EAGER` | — | Set to `true` to disable CUDA graphs (saves VRAM, slower) |
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | — | Override the model name returned by `/v1/models` |

Any [AsyncEngineArgs](https://docs.vllm.ai/en/latest/api/engine/async_llm_engine.html) field can be set via its `UPPERCASED` env var name.

### GPU sizing guide

| Model size | VRAM (BF16) | `TENSOR_PARALLEL_SIZE` | Example GPU |
|------------|-------------|------------------------|-------------|
| 1–3B | 3–8 GB | 1 | RTX 4090, RTX 5060 |
| 4–8B | 10–16 GB | 1 | RTX PRO 6000 (48 GB) |
| 26B | ~52 GB | 2 | 2× RTX PRO 6000 |
| 70B | ~140 GB | 4 | 4× A100 80 GB |
| 100B+ MoE | varies | 4–8 | 4–8× H100/A100 |

For 8 GB VRAM, use quantization to fit larger models:

```bash
-e QUANTIZATION=bitsandbytes \
-e LOAD_FORMAT=bitsandbytes \
-e ENFORCE_EAGER=true
```

---

## RunPod deployment

1. Create a serverless endpoint using image `gwesterrunpod/rp-gwestr-vllm:0.5.1`
2. Attach a network volume (50 GB+) mounted at `/runpod-volume`
3. Pre-download model weights to `/runpod-volume/models/<model-name>` on the volume
4. Set env vars — at minimum `MODEL_PATH=/runpod-volume/models/<model-name>`

The RunPod load balancer polls `/ping` and routes traffic once the worker returns 200.

---

## How it works

On startup, the worker:
1. Reads engine config from environment variables via `engine_args.py`
2. Initializes a vLLM `AsyncLLMEngine` in-process (no subprocess, no extra HTTP hop)
3. Builds OpenAI-compatible serving layers (`OpenAIServingChat`, `OpenAIServingCompletion`, etc.)
4. Starts a FastAPI/uvicorn server on port 80

All OpenAI-compatible clients work without modification — just point `base_url` at the worker.

---

## Inference backends

Built on [NVIDIA AI Dynamo](https://github.com/ai-dynamo/dynamo), which supports both vLLM and TensorRT-LLM. The current image uses vLLM for broad model compatibility.

| Backend | Strengths |
|---------|-----------|
| **vLLM** (current) | Any HuggingFace model, fast iteration, BF16/FP8, continuous batching |
| **TensorRT-LLM** | Maximum throughput, INT8/FP8, optimized CUDA kernels for production |

---

## Performance benchmarking

Install [aiperf](https://github.com/ai-dynamo/aiperf) (requires Python 3.12):

```bash
uv tool install aiperf --python 3.12
```

Run against a local server:

```bash
aiperf profile \
  --model google/gemma-3-1b-it \
  --tokenizer google/gemma-3-1b-it \
  --url http://localhost:8080 \
  --endpoint-type chat \
  --streaming \
  --concurrency 4 \
  --request-count 50
```

Run against a RunPod endpoint:

```bash
aiperf profile \
  --model google/gemma-3-1b-it \
  --tokenizer google/gemma-3-1b-it \
  --url https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai \
  --endpoint-type chat \
  --api-key $RUNPOD_API_KEY \
  --streaming \
  --concurrency 40 \
  --request-count 1000 \
  --public-dataset sharegpt
```

---

## Local tests

Unit tests (no GPU required):

```bash
pip install pytest pytest-asyncio
pytest tests/test_handler.py -v
```

Integration tests (requires running server and GPU):

```bash
VLLM_URL=http://localhost:8080 \
MODEL_PATH=google/gemma-3-1b-it \
AIPERF_CONCURRENCY=2 \
AIPERF_REQUESTS=10 \
pytest tests/test_integration.py -v -s
```

---

## Building and releasing

Bump `VERSION`, then:

```bash
./release.sh
```

This builds and pushes `gwesterrunpod/rp-gwestr-vllm:<version>` to Docker Hub. Pin the version tag in your RunPod endpoint template — avoid `latest` for reproducible deployments.

Published images: [hub.docker.com/r/gwesterrunpod/rp-gwestr-vllm](https://hub.docker.com/r/gwesterrunpod/rp-gwestr-vllm)

---

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
