# vLLM OpenAI server — serves any HF model with OpenAI-compatible chat API.
# Requires CUDA 12.1+ (driver 530+). vLLM 0.19.0 supports Gemma 4.
FROM vllm/vllm-openai:v0.19.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN pip3 install --no-cache-dir runpod httpx

WORKDIR /app
COPY handler.py .

# Defaults — all overridable at RunPod endpoint config time.
# MODEL_PATH must be provided at runtime pointing to model weights.
ENV VLLM_PORT=8000 \
    TENSOR_PARALLEL_SIZE=1 \
    MAX_MODEL_LEN=8192 \
    GPU_MEMORY_UTILIZATION=0.90 \
    MAX_NUM_SEQS=4 \
    TRITON_STARTUP_TIMEOUT=1800

ENTRYPOINT ["python3", "-u", "/app/handler.py"]
