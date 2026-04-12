# rp-gwestr-vllm

RunPod serverless endpoint using [vLLM](https://github.com/vllm-project/vllm) to serve any Hugging Face model with an OpenAI-compatible chat API.

Designed to scale from 4B → 8B → 26B → 100B+ models by adjusting env vars only — no code changes, no image rebuild.

---

## How it works

On worker startup, `handler.py`:
1. Launches `vllm serve` pointing at `MODEL_PATH`
2. Waits for `/health` to return 200
3. Forwards RunPod jobs to `/v1/chat/completions`

---

## Environment variables

Set these at the RunPod endpoint level (or via `docker run -e`). The image has no hardcoded model.

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | **required** | Path to model weights (e.g. `/runpod-volume/models/gemma-4-E4B-it`) |
| `TENSOR_PARALLEL_SIZE` | `1` | GPUs per worker — set to `2` for 26B, `4`+ for 70B+/100B+ |
| `MAX_MODEL_LEN` | `8192` | Max context window in tokens. Reduce if OOM on startup |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of VRAM vLLM may use for KV cache |
| `MAX_NUM_SEQS` | `4` | Max concurrent requests per worker |
| `VLLM_PORT` | `8000` | Internal port (no need to change) |
| `TRITON_STARTUP_TIMEOUT` | `1800` | Seconds to wait for vLLM ready — increase for large models |

### Sizing guide

| Model size | VRAM (BF16) | `TENSOR_PARALLEL_SIZE` | GPU |
|---|---|---|---|
| 4B (Gemma 4B) | ~16 GB | 1 | RTX PRO 4000 (24 GB) |
| 8B | ~16 GB | 1 | RTX PRO 4000 (24 GB) |
| 26B | ~52 GB | 2 | 2× RTX PRO 6000 (48 GB each) |
| 70B | ~140 GB | 4 | 4× A100 80 GB |
| 100B+ MoE | varies | 4–8 | 4–8× H100/A100 |

---

## Running locally

### Requirements

- NVIDIA driver **595.45+** (RTX PRO 4000/6000 Blackwell series)
- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### RTX PRO 4000 Blackwell (24 GB VRAM)

```bash
docker run --rm --gpus all \
  -v /path/to/model:/model:ro \
  -p 8000:8000 \
  -e MODEL_PATH=/model \
  -e MAX_MODEL_LEN=4096 \
  -e GPU_MEMORY_UTILIZATION=0.90 \
  -e MAX_NUM_SEQS=2 \
  gwesterrunpod/rp-gwestr-vllm:latest
```

### RTX PRO 6000 Blackwell (48 GB VRAM) — same driver, larger limits

```bash
docker run --rm --gpus all \
  -v /path/to/model:/model:ro \
  -p 8000:8000 \
  -e MODEL_PATH=/model \
  -e MAX_MODEL_LEN=8192 \
  -e GPU_MEMORY_UTILIZATION=0.92 \
  -e MAX_NUM_SEQS=8 \
  gwesterrunpod/rp-gwestr-vllm:latest
```

For a 26B model across **2× RTX PRO 6000** on the same host:

```bash
docker run --rm --gpus all \
  -v /path/to/26b-model:/model:ro \
  -p 8000:8000 \
  -e MODEL_PATH=/model \
  -e TENSOR_PARALLEL_SIZE=2 \
  -e MAX_MODEL_LEN=8192 \
  -e GPU_MEMORY_UTILIZATION=0.90 \
  -e MAX_NUM_SEQS=4 \
  gwesterrunpod/rp-gwestr-vllm:latest
```

### Test prompt

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [{"role": "user", "content": "How does RunPod serverless work?"}]
  }'
```

---

## Build and push

```bash
docker build -t gwesterrunpod/rp-gwestr-vllm:latest .
docker push gwesterrunpod/rp-gwestr-vllm:latest
```

---

## RunPod deployment

1. Push image to Docker Hub
2. Create a serverless endpoint:
   - **Image:** `gwesterrunpod/rp-gwestr-vllm:latest`
   - **Container disk:** 20 GB minimum
   - **Network volume:** mount at `/runpod-volume` (50 GB+)
3. Set env vars at the endpoint level — at minimum `MODEL_PATH`
4. Pre-load model weights onto the network volume before starting the endpoint

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
