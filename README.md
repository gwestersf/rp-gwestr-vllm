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
1. vLLM initializes in-process via the Python API, loading the model from `MODEL_PATH`
2. Incoming RunPod jobs are passed directly to the vLLM engine — no subprocess, no HTTP roundtrip
3. Responses are streamed or returned as a complete result

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

Any [AsyncEngineArgs](https://docs.vllm.ai/en/latest/api/engine/async_llm_engine.html) field can be set via its uppercased env var name — for example `ENFORCE_EAGER=true`, `QUANTIZATION=fp8`, `DTYPE=bfloat16`.

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

```bash
docker run --rm --gpus all \
  -v /path/to/model:/model:ro \
  -e MODEL_PATH=/model \
  -e MAX_MODEL_LEN=8192 \
  -e GPU_MEMORY_UTILIZATION=0.92 \
  -e MAX_NUM_SEQS=8 \
  gwesterrunpod/rp-gwestr-vllm:0.4.0
```

Multi-GPU:

```bash
docker run --rm --gpus all \
  -v /path/to/model:/model:ro \
  -e MODEL_PATH=/model \
  -e TENSOR_PARALLEL_SIZE=2 \
  -e MAX_MODEL_LEN=8192 \
  -e GPU_MEMORY_UTILIZATION=0.90 \
  -e MAX_NUM_SEQS=4 \
  gwesterrunpod/rp-gwestr-vllm:0.4.0
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

Send `"_run_tests": true` as the input to run a health check and chat sanity check directly inside the worker.

```json
{"input": {"_run_tests": true}}
```

With curl:

```bash
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{"input": {"_run_tests": true}}'
```

The response includes:
- `health` — engine initialization status
- `chat` — single completion sanity check and first 200 chars of response

---

## Performance benchmarking

Use [aiperf](https://github.com/ai-dynamo/aiperf) against the RunPod OpenAI-compatible endpoint for realistic load testing. RunPod exposes a standard OpenAI proxy at `https://api.runpod.ai/v2/{endpoint_id}/openai`.

```bash
aiperf profile \
  --model <model_name_or_path> \
  --tokenizer <local_tokenizer_path_or_hf_id> \
  --url https://api.runpod.ai/v2/{endpoint_id}/openai \
  --endpoint /v1/chat/completions \
  --api-key $RUNPOD_API_KEY \
  --endpoint-type chat \
  --concurrency 40 \
  --request-count 1000 \
  --streaming \
  --public-dataset sharegpt
```

If your tokenizer is a gated HuggingFace model, set `HF_TOKEN` before running.

Install aiperf locally with:
```bash
uv tool install aiperf --python 3.12
```

---

## Local testing

### Unit tests (no GPU required)

```bash
pip install pytest pytest-asyncio
pytest tests/test_handler.py -v
```

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
