# rp-tritonserver

A RunPod serverless worker that serves large language models via NVIDIA Triton Inference Server with the vLLM backend. Exposes an OpenAI-compatible chat completions interface. Model weights are loaded at runtime from a RunPod network volume — nothing is baked into the image.

> Generated with [Claude Code](https://claude.ai/claude-code)

---

## How it works

At container startup:

1. `generate_config.py` writes a Triton model repository to disk (`/model_repo` by default) using environment variables — `config.pbtxt` for the Triton backend config and `model.json` for the vLLM engine config.
2. `tritonserver` is launched pointing at that repository.
3. The model tokenizer is loaded from the network volume concurrently while Triton warms up.
4. Once Triton's health endpoint responds ready, the RunPod handler is registered and the worker begins accepting jobs.

Each job receives an OpenAI-style `messages` array, applies the model's chat template, sends the prompt to Triton's generate endpoint, and returns a response in OpenAI chat completions format. Streaming is supported via RunPod's async generator protocol.

---

## Base image

```
nvcr.io/nvidia/tritonserver:26.03-vllm-python-py3
```

Pull from [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver).

---

## Project structure

```
rp-tritonserver/
├── Dockerfile            # Extends the Triton vLLM image, adds runpod + httpx
├── generate_config.py    # Writes Triton model repo from env vars at startup
├── handler.py            # RunPod entry point — starts Triton, serves requests
├── pyproject.toml        # Project metadata and pytest config
└── tests/
    ├── test_generate_config.py
    └── test_handler.py
```

---

## Environment variables

All configuration is done via environment variables set in the RunPod endpoint UI. No values are hardcoded in the image.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | **required** | Absolute path to model weights on the network volume (e.g. `/runpod-volume/models/gemma-4-E4B-it`) |
| `MODEL_NAME` | `model` | Triton model name — used as the directory name inside the model repo and in API paths |
| `TRITON_MODEL_REPO` | `/model_repo` | Where the Triton model repository is generated at startup |
| `TRITON_HTTP_PORT` | `8000` | Port Triton listens on for HTTP |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism |
| `MAX_MODEL_LEN` | `8192` | Maximum context length (tokens) |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of GPU memory vLLM may use |
| `MAX_NUM_SEQS` | `4` | Maximum number of sequences processed concurrently |
| `TRITON_STARTUP_TIMEOUT` | `1800` | Seconds to wait for Triton to become ready (large models can take several minutes to load) |

---

## Build and push

```bash
cd rp-tritonserver

docker build -t gwesterrunpod/rp-tritonserver:26.03-vllm-python-py3 .
docker push gwesterrunpod/rp-tritonserver:26.03-vllm-python-py3
```

To target a different base version, override the build arg:

```bash
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/tritonserver:26.03-vllm-python-py3 \
  -t gwesterrunpod/rp-tritonserver:26.03-vllm-python-py3 .
```

---

## RunPod endpoint setup

1. **Create a serverless endpoint** in the RunPod console.
2. Set the container image to `gwesterrunpod/rp-tritonserver:26.03-vllm-python-py3`.
3. **Attach your network volume** containing the model weights.
4. Set at minimum the `MODEL_PATH` environment variable to the weights location on the volume.
5. Tune `TENSOR_PARALLEL_SIZE`, `MAX_MODEL_LEN`, and `GPU_MEMORY_UTILIZATION` for your GPU type and model.

---

## Request format

Jobs are submitted to the RunPod endpoint with the following input schema:

```json
{
  "input": {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain transformer attention in one paragraph."}
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": -1,
    "stop": [],
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stream": false
  }
}
```

`messages` is the only required field. All generation parameters are optional and fall back to defaults.

### Non-streaming response

```json
{
  "choices": [
    {
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ]
}
```

### Streaming response

Set `"stream": true`. The handler yields chunks in the same shape as OpenAI's streaming format:

```json
{"choices": [{"delta": {"role": "assistant", "content": "Attention"}, "finish_reason": null}]}
{"choices": [{"delta": {"role": "assistant", "content": " is"}, "finish_reason": null}]}
...
{"choices": [{"delta": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]}
```

---

## Running tests

Tests require no GPU, no Triton process, and no model weights. All external I/O is mocked.

```bash
pip install -e ".[test]"
pytest
```

The suite covers:

- `generate_config.py` — `config.pbtxt` content, `model.json` values, all env var overrides, defaults, custom model name, idempotency, missing `MODEL_PATH` error
- `handler.py` — sampling parameter mapping, response shaping, chat template application, non-streaming and streaming handler paths, init sequencing, startup timeout

One important implementation note reflected in the tests: the inner `generate()` async generator in the streaming path is **lazy** — it looks up `_stream` at iteration time, not when `handler()` returns. Tests that verify streaming behavior must iterate the generator while any relevant mock is still active.

---

## Switching models

Because all configuration is in environment variables, switching models requires only:

1. Download the new model weights to your network volume.
2. Update `MODEL_PATH` (and optionally `MODEL_NAME`, `MAX_MODEL_LEN`) in the RunPod endpoint config.
3. Redeploy — no image rebuild needed.
