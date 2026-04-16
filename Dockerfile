# vLLM load-balancer worker — native OpenAI-compatible API on port 80.
# Built on ai-dynamo/dynamo main (vLLM 0.19.0, CUDA 13.0, driver 575+).
#
# RunPod load-balancer polls GET /ping:
#   204 = engine loading (not yet in rotation)
#   200 = ready (accepting traffic)
#
# OpenAI-compatible endpoints:
#   POST /v1/chat/completions
#   POST /v1/completions
#   POST /v1/responses
#   POST /v1/messages  (Anthropic)
#
# MODEL_PATH must be provided at runtime. Any AsyncEngineArgs field can be
# set via its UPPERCASED env var name (see worker-vllm/src/engine_args.py).
FROM gwesterrunpod/dynamo-vllm-runtime:main

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN /opt/dynamo/venv/bin/pip install --no-cache-dir \
    runpod \
    fastapi \
    "uvicorn[standard]" \
    python-dotenv \
    "transformers==5.5.3" \
    "bitsandbytes>=0.48.1"

WORKDIR /app

# handler.py stub required by RunPod Hub tooling
COPY handler.py ./

# worker-vllm source provides vLLMEngine, tokenizer, utils
COPY worker-vllm/src /src
COPY handler_lb.py /src/handler_lb.py
# Override submodule engine_args.py: ours reads MODEL_PATH, theirs only reads MODEL_NAME
COPY engine_args.py /src/engine_args.py

# /  → 'from src.utils import …' resolves to /src/utils.py
# /src → sibling imports (engine, engine_args, tokenizer, utils, constants)
ENV PYTHONPATH="/:/src"

# HuggingFace cache on the network volume (mounted at /runpod-volume).
# MODEL_PATH can be a HF repo ID (e.g. meta-llama/Llama-3.1-8B-Instruct);
# vLLM will find the pre-downloaded model here instead of hitting the network.
ENV HF_HOME="/runpod-volume/huggingface-cache/hub" \
    HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub" \
    HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets"

# Defaults — all overridable at RunPod endpoint config time.
ENV TENSOR_PARALLEL_SIZE=1 \
    MAX_MODEL_LEN=8192 \
    GPU_MEMORY_UTILIZATION=0.90 \
    MAX_NUM_SEQS=4

EXPOSE 80

CMD ["/opt/dynamo/venv/bin/python3", "/src/handler_lb.py"]
