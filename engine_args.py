"""Build AsyncEngineArgs from environment variables.

Every AsyncEngineArgs field can be set via its UPPERCASED env var name.

    MAX_MODEL_LEN=4096          → max_model_len=4096
    TENSOR_PARALLEL_SIZE=2      → tensor_parallel_size=2
    GPU_MEMORY_UTILIZATION=0.9  → gpu_memory_utilization=0.9
    ENFORCE_EAGER=true          → enforce_eager=True
    QUANTIZATION=fp8            → quantization="fp8"

MODEL_PATH is required and maps to the `model` field.
"""

import dataclasses
import json
import logging
import os
import typing

from vllm import AsyncEngineArgs

_DEFAULTS = {
    "max_model_len": 8192,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 4,
    "trust_remote_code": True,
    "disable_log_stats": False,
}


def _coerce(value: str, field_type) -> object:
    """Convert a string env var value to the type expected by AsyncEngineArgs."""
    val = value.strip()
    if val in ("", "None", "none"):
        return None
    # Unwrap Optional[X] / Union[X, None]
    type_args = typing.get_args(field_type)
    if type_args:
        non_none = [a for a in type_args if a is not type(None)]
        if non_none:
            field_type = non_none[0]
    if field_type is bool:
        return val.lower() in ("true", "1", "yes")
    if field_type is int:
        return int(val)
    if field_type is float:
        return float(val)
    if field_type is str:
        return val
    origin = typing.get_origin(field_type)
    if origin in (dict, list):
        return json.loads(val)
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return val


def get_engine_args() -> AsyncEngineArgs:
    kwargs = dict(_DEFAULTS)
    kwargs["model"] = os.environ["MODEL_PATH"]

    try:
        hints = typing.get_type_hints(AsyncEngineArgs)
    except Exception:
        hints = {}

    for field in dataclasses.fields(AsyncEngineArgs):
        env_val = os.environ.get(field.name.upper())
        if env_val is None:
            continue
        try:
            kwargs[field.name] = _coerce(env_val, hints.get(field.name, str))
        except Exception as e:
            logging.warning("Skipping %s=%r: %s", field.name.upper(), env_val, e)

    return AsyncEngineArgs(**kwargs)
