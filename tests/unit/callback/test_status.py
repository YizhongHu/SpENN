"""Tests for line-oriented status callback output."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from spenn.callback import Event, Status, configure_terminal_logging
from tests.unit.callback.support import FakeState


def _context(tmp_path: Path) -> SimpleNamespace:
    metadata = SimpleNamespace(
        run_id="run-1",
        run_dir=str(tmp_path / "run-1"),
        device="cpu",
        dtype="float64",
        git_commit="abcdef123456",
        dirty_worktree=False,
        extra={
            "hardware": {
                "hostname": "node123",
                "cpu_count_logical": 64,
                "cpu_count_available": 8,
                "cuda_available": True,
                "cuda_device_count": 1,
                "cuda_devices": [
                    {
                        "index": 0,
                        "name": "NVIDIA A100-SXM4-40GB",
                        "total_memory_bytes": 40 * 1024**3,
                        "capability": "8.0",
                    }
                ],
            },
            "runtime": {
                "device": "cuda",
                "dtype": "float64",
                "python_version": "3.14.0",
                "torch_version": "2.9.0",
                "torch_cuda_version": "12.8",
                "cuda_visible_devices": "0",
            },
            "slurm": {
                "job_id": "123456",
                "array_task_id": "7",
                "cpus_per_task": "8",
                "job_partition": "kozinsky_gpu",
            },
        },
    )
    return SimpleNamespace(metadata=metadata)


def test_status_writes_json_and_terminal_lifecycle_lines(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    callback = Status(
        ["run_start", "run_end"],
        output_path=tmp_path / "status.json",
        color="never",
    )
    context = _context(tmp_path)

    with caplog.at_level(logging.INFO, logger="spenn.status"):
        callback.handle(Event(name="run_start", context=context))
        callback.handle(Event(name="run_end", context=context))

    messages = [record.getMessage() for record in caplog.records]
    assert any("SpENN Run Status" in message for message in messages)
    assert any("Hardware Environment" in message for message in messages)
    assert any("Run ID" in message and "run-1" in message for message in messages)
    assert any("Runtime Device" in message and "cuda" in message for message in messages)
    assert any("Torch CUDA" in message and "12.8" in message for message in messages)
    assert any("GPU 0 Name" in message and "NVIDIA A100-SXM4-40GB" in message for message in messages)
    assert any("GPU 0 Memory" in message and "40.0GB" in message for message in messages)
    assert any("SLURM Job ID" in message and "123456" in message for message in messages)
    assert any(message.startswith("[run] completed dir=") for message in messages)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "completed"


def test_status_renders_training_metrics_from_state(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    callback = Status(
        ["step_end"],
        terminal=True,
        color="never",
        include=[
            "train/loss",
            "train/energy",
            "train/sampler/acceptance_rate",
            "train/grad_norm",
            "train/local_energy_finite_fraction",
        ],
    )
    state = FakeState(
        step=10,
        metrics={
            "loss": 0.421,
            "energy": 2.104,
            "grad_norm": 0.012,
            "local_energy_finite_fraction": 1.0,
        },
        sampler_stats={"acceptance_rate": 0.61},
    )

    with caplog.at_level(logging.INFO, logger="spenn.status"):
        callback.handle(Event(name="step_end", context=_context(tmp_path), state=state, payload={"step": 10}))

    assert caplog.records[-1].getMessage() == (
        "[train] step=10 loss=0.421 energy=2.104 acc=0.61 grad=0.012 finite=1"
    )


def test_status_renders_evaluation_metrics(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    callback = Status(["evaluate_end"], terminal=True, color="never")

    with caplog.at_level(logging.INFO, logger="spenn.status"):
        callback.handle(
            Event(
                name="evaluate_end",
                context=_context(tmp_path),
                payload={"metrics": {"energy": 2.0, "energy_stderr": 0.01, "other": 3.0}},
            )
        )

    assert caplog.records[-1].getMessage() == "[eval] energy=2 stderr=0.01"


def test_status_terminal_false_suppresses_terminal_output(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    callback = Status(
        ["run_start"],
        output_path=tmp_path / "status.json",
        terminal=False,
        color="never",
    )

    with caplog.at_level(logging.INFO, logger="spenn.status"):
        callback.handle(Event(name="run_start", context=_context(tmp_path)))

    assert not caplog.records
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "running"


def test_configure_terminal_logging_adds_one_package_handler() -> None:
    logger_name = "spenn.test_terminal_status"
    logger = logging.getLogger(logger_name)
    original_handlers = list(logger.handlers)
    logger.handlers.clear()
    try:
        configure_terminal_logging(enabled=True, level="debug", color="never", logger_name=logger_name)
        configure_terminal_logging(enabled=True, level="info", color="never", logger_name=logger_name)

        handlers = [handler for handler in logger.handlers if getattr(handler, "_spenn_terminal_handler", False)]
        assert len(handlers) == 1
        assert handlers[0].level == logging.INFO
        assert logger.propagate is False
    finally:
        logger.handlers[:] = original_handlers
        logger.propagate = True


def test_status_rejects_invalid_color() -> None:
    with pytest.raises(ValueError, match="color"):
        Status(["run_start"], color="sometimes")
