"""Tests for run metadata provenance collection."""

from __future__ import annotations

from types import SimpleNamespace

import spenn.artifacts as artifacts


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
