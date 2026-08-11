"""Tests for line-oriented status callback output."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpen.callback import Event, Status, configure_terminal_logging
from tpen.callback.status import _format_status_box
from tests.unit.callback.support import (
    RecordingContext,
    deliver_completed_iteration,
    make_sampler_stats,
    training_state,
)


def _context(tmp_path: Path) -> SimpleNamespace:
    metadata = SimpleNamespace(
        run_id="run-1",
        run_name="unit-status",
        timestamp="2026-06-11T10:00:00-04:00",
        timezone="America/New_York",
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
    return SimpleNamespace(metadata=metadata, now_iso=lambda: "2026-06-11T10:00:00-04:00")


def test_status_writes_json_and_terminal_lifecycle_lines(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    callback = Status(output_path=tmp_path / "status.json", color="never")
    context = _context(tmp_path)

    with caplog.at_level(logging.INFO, logger="spenn.status"):
        callback.handle(Event(name="run_start", context=context))
        callback.handle(Event(name="run_end", context=context))

    messages = [record.getMessage() for record in caplog.records]
    assert any("TPEN Run Status" in message for message in messages)
    assert any("Hardware Environment" in message for message in messages)
    assert any("Run ID" in message and "run-1" in message for message in messages)
    assert any("Timezone" in message and "America/New_York" in message for message in messages)
    assert any("Started At" in message and "-04:00" in message for message in messages)
    assert any("Runtime Device" in message and "cuda" in message for message in messages)
    assert any("Torch CUDA" in message and "12.8" in message for message in messages)
    assert any("GPU 0 Name" in message and "NVIDIA A100-SXM4-40GB" in message for message in messages)
    assert any("GPU 0 Memory" in message and "40.0GB" in message for message in messages)
    assert any("SLURM Job ID" in message and "123456" in message for message in messages)
    assert any(message.startswith("[run] completed dir=") for message in messages)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["timezone"] == "America/New_York"
    assert status["start_time"] == "2026-06-11T10:00:00-04:00"
    assert status["end_time"] == "2026-06-11T10:00:00-04:00"


def _train_status(**overrides) -> Status:
    return Status(
        terminal=True,
        color="never",
        train_lines=True,
        include=[
            "train/loss",
            "train/energy",
            "train/sampler/acceptance_rate",
            "train/grad_norm",
            "train/local_energy_finite_fraction",
        ],
        **overrides,
    )


def _train_state():
    return training_state(
        step=10,
        metrics={
            "loss": 0.421,
            "energy": 2.104,
            "grad_norm": 0.012,
            "local_energy_finite_fraction": 1.0,
        },
        sampler_stats=make_sampler_stats(acceptance_rate=0.61),
    )


def test_status_renders_training_metrics_from_state(caplog: pytest.LogCaptureFixture) -> None:
    callback = _train_status()

    with caplog.at_level(logging.INFO, logger="spenn.status"):
        deliver_completed_iteration(callback, RecordingContext(), _train_state(), step=10)

    assert caplog.records[-1].getMessage() == (
        "[train] step=10 loss=0.421 energy=2.104 acc=0.61 grad=0.012 finite=1"
    )


def test_status_train_line_step_comes_from_the_event_not_the_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`TrainerState.step` is stale at any boundary above its assignment.

    The two coordinates are diverged here so the rendered one is unambiguous.
    """

    callback = _train_status()
    state = _train_state()
    state.step = 999

    with caplog.at_level(logging.INFO, logger="spenn.status"):
        deliver_completed_iteration(callback, RecordingContext(), state, step=10)

    assert caplog.records[-1].getMessage().startswith("[train] step=10 ")


def test_status_renders_no_train_line_by_default(caplog: pytest.LogCaptureFixture) -> None:
    """A run-lifecycle `Status` must not start narrating the training loop.

    Every shipped config builds `Status` for ``status.json`` alone. Subscribing
    it to completed iterations unconditionally would add one terminal line per
    step to every run, so the training line is a semantic option that replaces
    the ``triggers: [step_end]`` the old config path used to select it with.
    """

    callback = Status(color="never")

    with caplog.at_level(logging.INFO, logger="spenn.status"):
        deliver_completed_iteration(callback, RecordingContext(), _train_state(), step=10)

    assert not caplog.records


def test_status_terminal_false_suppresses_terminal_output(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    callback = Status(
        output_path=tmp_path / "status.json",
        terminal=False,
        color="never",
    )

    with caplog.at_level(logging.INFO, logger="spenn.status"):
        callback.handle(Event(name="run_start", context=_context(tmp_path)))

    assert not caplog.records
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "running"


def test_status_box_wraps_long_values_to_log_width() -> None:
    long_path = (
        "/n/holystore01/LABS/kozinsky_lab/Lab/User/rhu/TPEN/outputs/"
        "hooke_pair_smoke/pair/2026-06-11_142841_hooke_pair_smoke_train_7e9715/"
        "resolved_config.yaml"
    )

    lines = _format_status_box(
        "TPEN Run Status",
        [
            ("Run ID", "2026-06-11_142841_hooke_pair_smoke_train_7e9715"),
            ("Run Dir", long_path),
            ("Command", f"run.py --config {long_path}"),
        ],
    )

    assert all(len(line) <= 100 for line in lines)
    assert len(lines) > 7
    assert any(line.startswith("|        ") and " : " in line for line in lines)


def test_status_max_line_width_option_controls_start_boxes(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    callback = Status(color="never", max_line_width=72)
    context = _context(tmp_path)
    context.metadata.run_dir = (
        "/n/holystore01/LABS/kozinsky_lab/Lab/User/rhu/TPEN/outputs/"
        "hooke_pair_smoke/pair/run-1"
    )

    with caplog.at_level(logging.INFO, logger="spenn.status"):
        callback.handle(Event(name="run_start", context=context))

    box_lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith(("+", "|"))]
    assert box_lines
    assert all(len(line) <= 72 for line in box_lines)


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
        Status(color="sometimes")


def test_status_rejects_too_small_max_line_width() -> None:
    with pytest.raises(ValueError, match="max_line_width"):
        Status(max_line_width=39)


def test_status_keeps_only_the_three_run_level_legacy_triggers() -> None:
    """The residual legacy surface, pinned so it can only shrink deliberately.

    ``run_start``, ``run_end``, and ``exception`` are run-level events with no
    typed equivalent (item ``39eacd99``) and no owning domain, and they are the
    only thing that writes ``status.json``. They are hardcoded rather than
    configured because ADR-E002 forbids a config from naming events, and the
    training line that ``step_end`` used to select is now the ``train_lines``
    option.
    """

    callback = Status()

    assert callback.triggers == ("run_start", "run_end", "exception")
    assert not hasattr(callback, "on_step_end")
    assert not hasattr(callback, "on_evaluate_end")
