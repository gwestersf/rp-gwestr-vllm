# Default targets CUDA 12.x (driver 525+) for broadest RunPod compatibility.
# For CUDA 13.0 hardware (driver 595.45+, e.g. RTX Pro 6000), override at build time:
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/tritonserver:26.03-vllm-python-py3 .
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:25.09-vllm-python-py3

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN pip3 install --no-cache-dir runpod httpx "transformers>=4.51.0"

WORKDIR /app
COPY generate_config.py .
COPY handler.py .

# Defaults — all overridable at RunPod endpoint config time.
# MODEL_PATH is intentionally unset: must be provided at runtime
# pointing to the network volume location of your model weights.
ENV MODEL_NAME=model \
    TRITON_MODEL_REPO=/model_repo \
    TRITON_HTTP_PORT=8000 \
    TENSOR_PARALLEL_SIZE=1 \
    MAX_MODEL_LEN=8192 \
    GPU_MEMORY_UTILIZATION=0.90 \
    MAX_NUM_SEQS=4 \
    TRITON_STARTUP_TIMEOUT=1800

CMD ["python3", "-u", "/app/handler.py"]
