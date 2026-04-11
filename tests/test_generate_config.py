"""Tests for generate_config.py — no GPU or model weights required."""

import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def reload_module():
    """Re-import generate_config each test so module state is clean."""
    yield
    sys.modules.pop("generate_config", None)


def _import(tmp_path, monkeypatch, extra_env=None):
    """Helper: set minimal env, point repo at tmp_path, import the module."""
    monkeypatch.setenv("MODEL_PATH", "/vol/models/gemma")
    monkeypatch.setenv("TRITON_MODEL_REPO", str(tmp_path))
    for k, v in (extra_env or {}).items():
        monkeypatch.setenv(k, v)
    # clear optional vars so defaults are exercised unless overridden above
    for var in ["MODEL_NAME", "TENSOR_PARALLEL_SIZE", "MAX_MODEL_LEN",
                "GPU_MEMORY_UTILIZATION", "MAX_NUM_SEQS"]:
        if var not in (extra_env or {}):
            monkeypatch.delenv(var, raising=False)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    return importlib.import_module("generate_config")


# ── config.pbtxt ──────────────────────────────────────────────────────────────

class TestConfigPbtxt:
    def test_file_is_created(self, tmp_path, monkeypatch):
        mod = _import(tmp_path, monkeypatch)
        repo, name = mod.generate_model_repo()
        assert (Path(repo) / name / "config.pbtxt").exists()

    def test_backend_is_vllm(self, tmp_path, monkeypatch):
        mod = _import(tmp_path, monkeypatch)
        repo, name = mod.generate_model_repo()
        text = (Path(repo) / name / "config.pbtxt").read_text()
        assert 'backend: "vllm"' in text

    def test_decoupled_transaction_policy(self, tmp_path, monkeypatch):
        mod = _import(tmp_path, monkeypatch)
        repo, name = mod.generate_model_repo()
        text = (Path(repo) / name / "config.pbtxt").read_text()
        assert "decoupled: True" in text

    def test_required_inputs_present(self, tmp_path, monkeypatch):
        mod = _import(tmp_path, monkeypatch)
        repo, name = mod.generate_model_repo()
        text = (Path(repo) / name / "config.pbtxt").read_text()
        for field in ["text_input", "stream", "sampling_parameters",
                      "exclude_input_in_output"]:
            assert field in text, f"Missing input field: {field}"

    def test_output_field_present(self, tmp_path, monkeypatch):
        mod = _import(tmp_path, monkeypatch)
        repo, name = mod.generate_model_repo()
        text = (Path(repo) / name / "config.pbtxt").read_text()
        assert "text_output" in text


# ── model.json ────────────────────────────────────────────────────────────────

class TestModelJson:
    def _load(self, tmp_path, monkeypatch, extra=None):
        mod = _import(tmp_path, monkeypatch, extra)
        repo, name = mod.generate_model_repo()
        path = Path(repo) / name / "1" / "model.json"
        assert path.exists()
        return json.loads(path.read_text())

    def test_model_path_written(self, tmp_path, monkeypatch):
        data = self._load(tmp_path, monkeypatch)
        assert data["model"] == "/vol/models/gemma"

    def test_default_tensor_parallel(self, tmp_path, monkeypatch):
        assert self._load(tmp_path, monkeypatch)["tensor_parallel_size"] == 1

    def test_default_max_model_len(self, tmp_path, monkeypatch):
        assert self._load(tmp_path, monkeypatch)["max_model_len"] == 8192

    def test_default_gpu_memory_utilization(self, tmp_path, monkeypatch):
        assert self._load(tmp_path, monkeypatch)["gpu_memory_utilization"] == 0.90

    def test_default_max_num_seqs(self, tmp_path, monkeypatch):
        assert self._load(tmp_path, monkeypatch)["max_num_seqs"] == 4

    def test_overrides_applied(self, tmp_path, monkeypatch):
        data = self._load(tmp_path, monkeypatch, {
            "TENSOR_PARALLEL_SIZE": "4",
            "MAX_MODEL_LEN": "4096",
            "GPU_MEMORY_UTILIZATION": "0.80",
            "MAX_NUM_SEQS": "16",
        })
        assert data["tensor_parallel_size"] == 4
        assert data["max_model_len"] == 4096
        assert data["gpu_memory_utilization"] == 0.80
        assert data["max_num_seqs"] == 16

    def test_disable_log_requests_default_true(self, tmp_path, monkeypatch):
        assert self._load(tmp_path, monkeypatch)["disable_log_requests"] is True


# ── Return values & naming ────────────────────────────────────────────────────

class TestReturnValues:
    def test_default_model_name(self, tmp_path, monkeypatch):
        mod = _import(tmp_path, monkeypatch)
        _, name = mod.generate_model_repo()
        assert name == "model"

    def test_custom_model_name(self, tmp_path, monkeypatch):
        mod = _import(tmp_path, monkeypatch, {"MODEL_NAME": "gemma4"})
        repo, name = mod.generate_model_repo()
        assert name == "gemma4"
        assert (Path(repo) / "gemma4" / "config.pbtxt").exists()

    def test_returns_repo_path(self, tmp_path, monkeypatch):
        mod = _import(tmp_path, monkeypatch)
        repo, _ = mod.generate_model_repo()
        assert repo == str(tmp_path)

    def test_idempotent(self, tmp_path, monkeypatch):
        """Calling twice should not raise (directories already exist)."""
        mod = _import(tmp_path, monkeypatch)
        mod.generate_model_repo()
        mod.generate_model_repo()


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrors:
    def test_missing_model_path_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MODEL_PATH", raising=False)
        monkeypatch.setenv("TRITON_MODEL_REPO", str(tmp_path))
        sys.path.insert(0, str(Path(__file__).parent.parent))
        mod = importlib.import_module("generate_config")
        with pytest.raises(ValueError, match="MODEL_PATH"):
            mod.generate_model_repo()
