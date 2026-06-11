"""Tests for run metadata provenance collection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from omegaconf import DictConfig, OmegaConf

import spenn.artifacts as artifacts
from spenn.run import prepare_run_context


class FakeCuda:
    """Small fake ``torch.cuda`` surface for metadata tests."""

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def get_device_properties(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(
            name="NVIDIA A100-SXM4-40GB",
            total_memory=40 * 1024**3,
            major=8,
            minor=0,
        )


class FakeTorch:
    """Small fake torch module for metadata tests."""

    __version__ = "2.9.0"
    version = SimpleNamespace(cuda="12.8")
    cuda = FakeCuda()


class FakePartialTorch:
    """Torch-like module missing optional CUDA and NN surfaces."""

    __version__ = "partial"


def test_collect_hardware_metadata_records_runtime_hardware_and_slurm(
    monkeypatch,
) -> None:
    monkeypatch.setattr(artifacts, "import_module", lambda name: FakeTorch)
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "kozinsky_gpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    metadata = artifacts.collect_hardware_metadata(device="cuda", dtype="float64")

    assert metadata["runtime"]["device"] == "cuda"
    assert metadata["runtime"]["dtype"] == "float64"
    assert metadata["runtime"]["torch_version"] == "2.9.0"
    assert metadata["runtime"]["torch_cuda_version"] == "12.8"
    assert metadata["runtime"]["cuda_visible_devices"] == "0"
    assert metadata["hardware"]["hostname"]
    assert metadata["hardware"]["cpu_count_logical"] is not None
    assert metadata["hardware"]["cuda_available"] is True
    assert metadata["hardware"]["cuda_device_count"] == 1
    assert metadata["hardware"]["cuda_devices"] == [
        {
            "index": 0,
            "name": "NVIDIA A100-SXM4-40GB",
            "total_memory_bytes": 40 * 1024**3,
            "capability": "8.0",
        }
    ]
    for key, value in {
        "job_id": "123456",
        "array_task_id": "7",
        "cpus_per_task": "8",
        "job_partition": "kozinsky_gpu",
    }.items():
        assert metadata["slurm"][key] == value


def test_collect_hardware_metadata_tolerates_partial_torch(monkeypatch) -> None:
    monkeypatch.setattr(artifacts, "import_module", lambda name: FakePartialTorch)

    metadata = artifacts.collect_hardware_metadata(device="cpu", dtype="float64")

    assert metadata["runtime"]["torch_version"] == "partial"
    assert metadata["runtime"]["torch_cuda_version"] is None
    assert metadata["hardware"]["cuda_available"] is False
    assert metadata["hardware"]["cuda_device_count"] == 0
    assert metadata["hardware"]["cuda_devices"] == []


def test_prepare_run_context_defaults_to_utc_timezone(tmp_path: Path) -> None:
    context = prepare_run_context(_run_cfg(tmp_path), config_path="test.yaml", command="pytest")

    assert context.clock.timezone == "UTC"
    assert context.cfg.run.timezone == "UTC"
    assert context.source_cfg.run.timezone == "UTC"
    assert context.metadata.timezone == "UTC"
    assert datetime.fromisoformat(context.metadata.timestamp).utcoffset().total_seconds() == 0.0


def test_prepare_run_context_enforces_configured_timezone(tmp_path: Path) -> None:
    cfg = _run_cfg(tmp_path)
    cfg.run.timezone = "America/New_York"

    context = prepare_run_context(cfg, config_path="test.yaml", command="pytest")

    timestamp = datetime.fromisoformat(context.metadata.timestamp)
    assert context.clock.timezone == "America/New_York"
    assert context.cfg.run.timezone == "America/New_York"
    assert context.metadata.timezone == "America/New_York"
    assert timestamp.utcoffset() == ZoneInfo("America/New_York").utcoffset(timestamp)


def test_prepare_run_context_rejects_invalid_timezone(tmp_path: Path) -> None:
    cfg = _run_cfg(tmp_path)
    cfg.run.timezone = "Mars/Base"

    with pytest.raises(ValueError, match="run.timezone"):
        prepare_run_context(cfg, config_path="test.yaml", command="pytest")


def _run_cfg(tmp_path: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "experiment": {"name": "metadata", "sector": "unit", "run_name": "metadata_unit"},
            "run": {"root": str(tmp_path), "run_id": None, "dir": None},
            "runtime": {"device": "cpu", "dtype": "float64"},
            "callbacks": [],
            "loggers": [],
        }
    )
