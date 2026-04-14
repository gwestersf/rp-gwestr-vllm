# vLLM OpenAI server — serves any HF model with OpenAI-compatible chat API.
# Built on ai-dynamo/dynamo main (vLLM 0.19.0, CUDA 13.0, driver 575+).
FROM gwesterrunpod/dynamo-vllm-runtime:main

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN /opt/dynamo/venv/bin/pip install --no-cache-dir runpod httpx aiperf \
 && /opt/dynamo/venv/bin/pip install --no-cache-dir "transformers==5.5.3"

WORKDIR /app
COPY handler.py .

# Defaults — all overridable at RunPod endpoint config time.
# MODEL_PATH must be provided at runtime pointing to model weights.
ENV VLLM_PORT=8000 \
    TENSOR_PARALLEL_SIZE=1 \
    MAX_MODEL_LEN=8192 \
    GPU_MEMORY_UTILIZATION=0.90 \
    MAX_NUM_SEQS=4 \
    VLLM_STARTUP_TIMEOUT=1800

ENTRYPOINT ["python3", "-u", "/app/handler.py"]
