"""Generates the Triton model repository structure from environment variables.

Run standalone to validate config generation:
    MODEL_PATH=/runpod-volume/models/gemma-4 python3 generate_config.py
"""

import json
import os
import pathlib


def generate_model_repo() -> tuple[str, str]:
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise ValueError("MODEL_PATH environment variable is required")

    model_name = os.environ.get("MODEL_NAME", "model")
    triton_repo = os.environ.get("TRITON_MODEL_REPO", "/model_repo")
    tensor_parallel = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "8192"))
    gpu_memory_util = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
    max_num_seqs = int(os.environ.get("MAX_NUM_SEQS", "4"))

    version_dir = pathlib.Path(triton_repo) / model_name / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    config_text = f"""\
backend: "vllm"
max_batch_size: 0

model_transaction_policy {{
  decoupled: True
}}

input [
  {{
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }},
  {{
    name: "stream"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "sampling_parameters"
    data_type: TYPE_STRING
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "exclude_input_in_output"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    optional: true
  }}
]

output [
  {{
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }}
]

instance_group [
  {{
    kind: KIND_MODEL
  }}
]
"""

    (pathlib.Path(triton_repo) / model_name / "config.pbtxt").write_text(config_text)

    vllm_config = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_util,
        "max_num_seqs": max_num_seqs,
        "disable_log_requests": True,
        "enforce_eager": False,
    }
    (version_dir / "model.json").write_text(json.dumps(vllm_config, indent=2))

    print(f"[config] repo={triton_repo}  model={model_name}  path={model_path}")
    return triton_repo, model_name


if __name__ == "__main__":
    generate_model_repo()
