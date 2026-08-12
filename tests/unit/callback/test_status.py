"""Tests for line-oriented status callback output."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpen.callback import Status, configure_terminal_logging
from tpen.callback.status import _format_status_box
from tpen.events import Occurrence
from tpen.run_events import RunCompleted, RunFailed, RunStarted
from tests.unit.callback.support import (
    RecordingContext,
    deliver_completed_iteration,
    make_sampler_stats,
    training_state,
)


class _StatusContext:
    """Small direct-handler context implementing ``CallbackContext``."""

    def __init__(self, metadata: SimpleNamespace) -> None:
        self.metadata = metadata

    def now_iso(self) -> str:
        return "2026-06-11T10:00:00-04:00"


def _context(tmp_path: Path) -> _StatusContext:
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
    return _StatusContext(metadata)


def _deliver_run_event(callback: Status, context, event) -> None:
    """Hand one run-lifecycle occurrence to ``callback``'s stateless route.

    ``state=None``, because that is what the run lifecycle carries and what
    `tpen.artifacts.RunContext._dispatch_occurrence` passes for it. Delivering
    the same occurrence through the REAL dispatcher, rather than through this
    context stand-in, is covered in ``test_typed_run_lifecycle.py``; here the
    stand-in is what lets these tests assert rendered box content without
    building a run directory.
    """

    callback.handle_occurrence(Occurrence(event=event, count=1), context, None)


def test_status_writes_json_and_terminal_lifecycle_lines(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    callback = Status(output_path=tmp_path / "status.json", color="never")
    context = _context(tmp_path)

    with caplog.at_level(logging.INFO, logger="spenn.status"):
        _deliver_run_event(callback, context, RunStarted())
        _deliver_run_event(callback, context, RunCompleted())

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
        _deliver_run_event(callback, _context(tmp_path), RunStarted())

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
        _deliver_run_event(callback, context, RunStarted())

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


def test_status_records_a_failed_run_from_the_failure_boundary(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """THE failure this migration could most easily have shipped in silence.

    ``run_end`` and ``exception`` were two strings, and a migration that mapped
    both onto `RunCompleted` -- or that simply forgot `RunFailed` while adding
    the other two selectors -- would leave every test that emits a SUCCESS path
    green while ``status.json`` silently stopped recording failed runs. Runs
    fail on a cluster far more often than anyone reads the passing tests, so
    that regression would have been found by a missing artifact months later.

    Nothing but `RunFailed` is delivered here, so the assertion cannot be
    satisfied by a completion path.
    """

    callback = Status(output_path=tmp_path / "status.json", color="never")
    context = _context(tmp_path)

    with caplog.at_level(logging.INFO, logger="spenn.status"):
        _deliver_run_event(callback, context, RunStarted())
        _deliver_run_event(
            callback,
            context,
            RunFailed(exception_type="ValueError", exception_message="boom in the loop"),
        )

    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["exception_type"] == "ValueError"
    assert status["exception_message"] == "boom in the loop"
    assert status["end_time"] == "2026-06-11T10:00:00-04:00"
    # ``current_event`` names a moment in a durable artifact rather than
    # selecting an event, so it keeps the value shipped runs already carry.
    assert status["current_event"] == "exception"
    assert caplog.records[-1].getMessage() == (
        f"[run] failed dir={context.metadata.run_dir} "
        'exception=ValueError message="boom in the loop"'
    )


def test_status_writes_the_three_statuses_on_the_three_run_boundaries(tmp_path: Path) -> None:
    """One boundary per status, and the ``current_event`` value each carries."""

    callback = Status(output_path=tmp_path / "status.json", color="never", terminal=False)
    context = _context(tmp_path)
    written = []

    for event in (
        RunStarted(),
        RunCompleted(),
        RunFailed(exception_type="RuntimeError", exception_message="late"),
    ):
        _deliver_run_event(callback, context, event)
        written.append(json.loads((tmp_path / "status.json").read_text()))

    assert [entry["status"] for entry in written] == ["running", "completed", "failed"]
    assert [entry["current_event"] for entry in written] == [
        "run_start",
        "run_end",
        "exception",
    ]
    assert written[0]["end_time"] is None
    assert [entry["start_time"] for entry in written] == ["2026-06-11T10:00:00-04:00"] * 3


def test_status_writes_nothing_for_the_legacy_run_level_strings(tmp_path: Path) -> None:
    """The migration MOVED the write; it did not add a second one.

    A callback that kept its ``on_run_end`` while gaining the typed group would
    write ``status.json`` twice per boundary and pass every other test in this
    file. `_CallbackCore.handle` dispatches by name, so the check is that the
    name resolves to nothing.
    """

    from tpen.callback import Event

    callback = Status(output_path=tmp_path / "status.json", color="never")
    context = _context(tmp_path)

    for name in ("run_start", "run_end", "run_failed", "exception"):
        callback.handle(Event(name=name, context=context))

    assert not (tmp_path / "status.json").exists()


def test_status_subscribes_the_run_lifecycle_with_the_training_line_off(
    tmp_path: Path,
) -> None:
    """The shipped default writes ``status.json``, and its plan is all-stateless.

    ``Status(train_lines=False)`` is what every shipped config builds, and after
    this migration its ONLY subscription group is the state-free run-lifecycle
    one. `StatefulCallback._validate_typed_groups` refuses an all-stateless plan
    on a class that can never route its ``state_type``; this class can, through
    ``handle_occurrence_impl``, so the instance must construct and be delivered
    to. Both halves are asserted, because a construction that succeeded while
    delivering nothing is the exact silence this callback keeps falling into.
    """

    callback = Status(output_path=tmp_path / "status.json", color="never", terminal=False)

    groups = [state.group for state in callback._typed_group_states]
    assert [group.stateless for group in groups] == [True]

    _deliver_run_event(callback, _context(tmp_path), RunStarted())
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "running"


def test_status_keeps_its_domain_group_and_its_run_group_apart(tmp_path: Path) -> None:
    """``train_lines=True`` adds a group; it does not replace the run-level one.

    Both routes on one instance is the whole point of the ADR-E008 amendment,
    and the failure mode is one group quietly shadowing the other.
    """

    callback = Status(
        output_path=tmp_path / "status.json",
        color="never",
        terminal=False,
        train_lines=True,
    )

    groups = [state.group for state in callback._typed_group_states]
    assert [group.stateless for group in groups] == [False, True]

    _deliver_run_event(callback, _context(tmp_path), RunCompleted())
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "completed"

    deliver_completed_iteration(callback, RecordingContext(), _train_state(), step=10)
