"""Terminal and artifact status callbacks."""

from __future__ import annotations

import json
import logging
import socket
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from tpen.artifacts import RunContext, write_json
from tpen.events import DomainState
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.training.events import TrainingIterationCompleted
from tpen.training.state import TrainerState

from .base import Event, StatefulCallback
from .cadence import StepCadenceGate, SubscriptionGroup, pop_step_cadence
from .terminal_logging import color_status_line, validate_terminal_choice

_STATUS_BOX_MAX_LINE_WIDTH = 100
_STATUS_BOX_BORDER_WIDTH = 4
_STATUS_BOX_SEPARATOR = " : "


class Status(StatefulCallback[TrainerState]):
    """Write lifecycle status artifacts and terminal status lines.

    Parameters
    ----------
    output_path : str or pathlib.Path or None, optional
        Where the run's ``status.json`` is written. ``None`` writes no file.
    terminal : bool, optional
        Whether any line is emitted to the terminal logging channel at all.
    logger_name : str, optional
        Logger the status lines are written to.
    include : sequence of str, optional
        Fully-qualified metric identities rendered on the per-iteration training
        line, in order.
    color : {"auto", "always", "never"}, optional
        Terminal colouring policy.
    max_line_width : int, optional
        Width the ``run_start`` status boxes are wrapped to.
    train_lines : bool, optional
        Whether to render one compact line per completed training iteration.
        Off by default, so a `Status` configured only for the run lifecycle does
        not start narrating the loop.
    **kwargs
        Forwarded to `StatefulCallback` (e.g. ``every_n_steps``).

    Notes
    -----
    This callback is HALF migrated, deliberately. The per-iteration training
    line reads `tpen.training.state.TrainerState` and now arrives through typed
    delivery at `tpen.training.events.TrainingIterationCompleted`.

    Three legacy string triggers survive: ``run_start``, ``run_end``, and
    ``exception``. They are the only thing that writes ``status.json`` at all,
    so dropping them would delete the run's status artifact outright.

    THE REASON THEY SURVIVE HAS CHANGED, and the new one is a harder blocker
    than the old. Until item ``39eacd99`` these three had no typed equivalent;
    they now do (`tpen.run_events`). They are still here because this class is a
    `StatefulCallback`. `tpen.artifacts.RunContext._dispatch_occurrence` decides
    delivery by ``isinstance(state, callback.state_type)``, and the run lifecycle
    carries no domain state, so a run-level occurrence is SKIPPED for this
    callback -- silently, exactly as
    ``test_a_boundary_emitted_without_state_delivers_nothing`` pins. Subscribing
    them here would not half-migrate the callback; it would stop ``status.json``
    being written, with no error anywhere.

    A callback declares ONE ``state_type``, so this is not fixable inside the
    class. It needs either a way for a subscription group to declare that it
    needs no state, or this class split into a run-lifecycle `Callback` and a
    training-line `StatefulCallback` (the shape ADR-E002 names for divergent
    policy modes). The first changes the ADR-E008 mechanism, the second changes a
    config-facing ``_target_`` contract, so both belong to the mechanism owner
    (``62593af4``) rather than to the run-lifecycle slice. The identical
    situation, for the identical reason, holds for the residual ``run_end`` on
    `tpen.callback.ArtifactIndex`; those two are now the whole remaining blocker
    for the residual string deletion (``85870732``).

    They are hardcoded rather than configured because every shipped config
    listed exactly these three, and because ADR-E002 forbids a config from
    naming the events a callback answers to. ``train_lines`` replaces the
    ``triggers: [step_end]`` spelling that selected the training line, as a
    semantic option rather than an event selector.

    One capability was lost in the migration and is recorded here rather than
    silently dropped. The training line used to also render metrics that other
    CALLBACKS attached to the shared legacy ``step_end`` payload -- in practice
    ``train/perf/step_time_sec`` and ``train/perf/step_time_sec_rolling_mean``
    from `tpen.callback.timing.TrainStepTiming`. That inter-callback side
    channel is the untyped payload dict ADR-E007 rejects, and a typed occurrence
    has no payload to replace it with, so those two keys cannot currently
    render. They stay in `_DEFAULT_STATUS_METRICS` because they name real
    published metrics that a typed route would restore; the metrics themselves
    are unaffected, since ``TrainStepTiming`` still logs them to every
    configured logger. No shipped config renders a training line today, so no
    run's terminal output changes.
    """

    # ClassVar: the runtime authority for typed state delivery.
    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(
        self,
        output_path: str | Path | None = None,
        *,
        terminal: bool = True,
        logger_name: str = "spenn.status",
        include: Sequence[str] | None = None,
        color: str = "auto",
        max_line_width: int = _STATUS_BOX_MAX_LINE_WIDTH,
        train_lines: bool = False,
        **kwargs: Any,
    ) -> None:
        cadence = pop_step_cadence(kwargs)
        super().__init__(
            # See the class notes: run-level residual, not a configurable knob.
            triggers=("run_start", "run_end", "exception"),
            # An instance rendering no training line subscribes to nothing,
            # rather than subscribing and discarding every delivery.
            typed_groups=(
                (
                    SubscriptionGroup(
                        selectors=(Subscription.of(TrainingIterationCompleted),)
                    ),
                )
                if train_lines
                else ()
            ),
            **kwargs,
        )
        self.output_path = None if output_path is None else Path(output_path)
        self.terminal = bool(terminal)
        self.train_lines = bool(train_lines)
        self.logger = logging.getLogger(logger_name)
        self.include = tuple(_DEFAULT_STATUS_METRICS if include is None else include)
        self.color = validate_terminal_choice(color, name="color")
        self.max_line_width = _validate_max_line_width(max_line_width)
        # The scheduling scalars gate the typed training line only. On the
        # legacy path they could never have gated the three run-level triggers
        # anyway: those carry no step, and `_CallbackCore.should_run` rejected a
        # `None` coordinate outright the moment `every_n_steps` was set.
        self._steps = StepCadenceGate(cadence)
        self.start_time: str | None = None

    def on_run_start(self, event: Event) -> None:
        """Record run start."""

        self.start_time = _now(event)
        for line in _format_run_start_lines(event, max_line_width=self.max_line_width):
            self._log_status(line, kind="run")
        self._write(
            event,
            status="running",
            current_event=event.name,
            end_time=None,
            exception_type=None,
            exception_message=None,
        )

    def on_run_end(self, event: Event) -> None:
        """Record successful completion."""

        self._log_status(_format_run_end(event), kind="completed")
        self._write(
            event,
            status="completed",
            current_event=event.name,
            end_time=_now(event),
            exception_type=None,
            exception_message=None,
        )

    def on_exception(self, event: Event) -> None:
        """Record run failure."""

        exception = event.payload.get("exception")
        self._log_status(_format_run_failure(event, exception), kind="failed")
        self._write(
            event,
            status="failed",
            current_event=event.name,
            end_time=_now(event),
            exception_type=None if exception is None else type(exception).__name__,
            exception_message=None if exception is None else str(exception),
        )

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: TrainerState,
    ) -> None:
        """Write one compact training status line for a completed iteration."""

        del context
        event = occurrence.event
        if not isinstance(event, TrainingIterationCompleted):
            return
        # The coordinate rides the typed event, never `state.step`, whose value
        # fields are stale at any boundary above their assignment.
        step = int(event.iteration.step)
        if not self._steps.should_run(step):
            return
        line = _format_train_status(state, self.include, step=step)
        if line is not None:
            self._log_status(line, kind="train")

    def _log_status(self, line: str, *, kind: str) -> None:
        if not self.terminal:
            return
        self.logger.info(color_status_line(line, kind=kind, color=self.color))

    def _write(
        self,
        event: Event,
        *,
        status: str,
        current_event: str,
        end_time: str | None,
        exception_type: str | None,
        exception_message: str | None,
    ) -> None:
        if self.output_path is None:
            return
        write_json(
            self.output_path,
            {
                "status": status,
                "timezone": event.context.metadata.timezone,
                "start_time": self.start_time,
                "end_time": end_time,
                "current_event": current_event,
                "exception_type": exception_type,
                "exception_message": exception_message,
            },
        )


def _now(event: Event) -> str:
    return event.context.now_iso()


_DEFAULT_STATUS_METRICS = (
    "train/loss",
    "train/energy",
    "train/energy_stderr",
    "train/sampler/acceptance_rate",
    "train/grad_norm",
    "train/local_energy_finite_fraction",
    # The two `train/perf` identities below name real published metrics but
    # cannot currently render: they reached this callback only through the
    # legacy `step_end` payload another callback mutated in place, and a typed
    # occurrence has no payload. See the note on `Status`.
    "train/perf/step_time_sec",
    "train/perf/step_time_sec_rolling_mean",
)

_STATUS_LABELS = {
    "train/loss": "loss",
    "train/energy": "energy",
    "train/energy_stderr": "stderr",
    "train/sampler/acceptance_rate": "acc",
    "train/grad_norm": "grad",
    "train/local_energy_finite_fraction": "finite",
    "train/perf/step_time_sec": "step_time",
    "train/perf/step_time_sec_rolling_mean": "step_avg",
}

def _format_run_start_lines(event: Event, *, max_line_width: int = _STATUS_BOX_MAX_LINE_WIDTH) -> list[str]:
    metadata = event.context.metadata
    extra = getattr(metadata, "extra", {}) or {}
    hardware = extra.get("hardware") if isinstance(extra, Mapping) else None
    runtime = extra.get("runtime") if isinstance(extra, Mapping) else None
    slurm = extra.get("slurm") if isinstance(extra, Mapping) else None
    status_rows: list[tuple[str, object] | None] = [
        ("Run ID", metadata.run_id),
        ("Run Dir", metadata.run_dir),
        ("Run Name", getattr(metadata, "run_name", "")),
        ("Timezone", getattr(metadata, "timezone", "")),
        ("Started At", getattr(metadata, "timestamp", "")),
        ("Status", "starting"),
        None,
        ("Git Commit", metadata.git_commit[:7] if metadata.git_commit else ""),
        ("Git Branch", getattr(metadata, "git_branch", "")),
        ("Dirty Worktree", metadata.dirty_worktree),
    ]
    command = getattr(metadata, "command", None)
    if command:
        status_rows.append(("Command", command))
    config_path = getattr(metadata, "config_path", None)
    if config_path:
        status_rows.append(("Config", config_path))

    hardware_rows: list[tuple[str, object] | None] = []
    if isinstance(runtime, Mapping):
        hardware_rows.extend(
            [
                ("Runtime Device", runtime.get("device", metadata.device)),
                ("Runtime DType", runtime.get("dtype", metadata.dtype)),
                ("Python", runtime.get("python_version", "unknown")),
                ("Torch", runtime.get("torch_version", "unavailable")),
                ("Torch CUDA", runtime.get("torch_cuda_version") or "unavailable"),
            ]
        )
        if runtime.get("cuda_visible_devices"):
            hardware_rows.append(("CUDA_VISIBLE_DEVICES", runtime["cuda_visible_devices"]))
    else:
        hardware_rows.extend([("Runtime Device", metadata.device), ("Runtime DType", metadata.dtype)])
    hardware_rows.append(None)

    if isinstance(hardware, Mapping):
        hardware_rows.extend(
            [
                ("Host", hardware.get("hostname", socket.gethostname())),
                ("Platform", hardware.get("platform", "unknown")),
                ("Machine", hardware.get("machine", "unknown")),
                ("Logical CPUs", hardware.get("cpu_count_logical")),
                ("Available CPUs", hardware.get("cpu_count_available")),
                ("CUDA Available", hardware.get("cuda_available", False)),
                ("CUDA Device Count", hardware.get("cuda_device_count", 0)),
            ]
        )
        devices = hardware.get("cuda_devices")
        if isinstance(devices, Sequence) and not isinstance(devices, str):
            for device in devices:
                if not isinstance(device, Mapping):
                    continue
                index = device.get("index")
                memory = (
                    _format_gib(device["total_memory_bytes"])
                    if isinstance(device.get("total_memory_bytes"), int | float)
                    else "unknown"
                )
                hardware_rows.extend(
                    [
                        (f"GPU {index} Name", device.get("name", "unknown")),
                        (f"GPU {index} Memory", memory),
                        (f"GPU {index} Capability", device.get("capability", "unknown")),
                    ]
                )
                if device.get("error"):
                    hardware_rows.append((f"GPU {index} Error", device["error"]))
    else:
        hardware_rows.append(("Host", socket.gethostname()))

    if isinstance(slurm, Mapping) and slurm:
        hardware_rows.append(None)
        for key, label in (
            ("job_id", "SLURM Job ID"),
            ("array_task_id", "SLURM Array Task"),
            ("cpus_per_task", "SLURM CPUs/Task"),
            ("mem_per_node", "SLURM Mem/Node"),
            ("job_partition", "SLURM Partition"),
            ("job_name", "SLURM Job Name"),
        ):
            if key in slurm:
                hardware_rows.append((label, slurm[key]))
    return [
        *_format_status_box("TPEN Run Status", status_rows, max_line_width=max_line_width),
        *_format_status_box("Hardware Environment", hardware_rows, max_line_width=max_line_width),
    ]


def _format_run_end(event: Event) -> str:
    return f"[run] completed dir={event.context.metadata.run_dir}"


def _format_run_failure(event: Event, exception: object | None) -> str:
    parts = ["[run] failed", f"dir={event.context.metadata.run_dir}"]
    if exception is not None:
        parts.extend([f"exception={type(exception).__name__}", f"message={_quote_value(str(exception))}"])
    return " ".join(parts)


def _format_train_status(
    state: TrainerState, include: Sequence[str], *, step: int
) -> str | None:
    values = _training_metric_values(state)
    rendered = [
        f"{_STATUS_LABELS.get(identity, identity)}={_format_status_value(values[identity])}"
        for identity in include
        if identity in values
    ]
    if not rendered:
        return None
    return " ".join([f"[train] step={step}", *rendered])


def _training_metric_values(state: TrainerState) -> dict[str, object]:
    """Compose the metric identities reachable from typed training state.

    Both fields are read by name off a typed `tpen.training.state.TrainerState`,
    not probed: ``None`` and an empty mapping are declared values, not absent
    attributes.
    """

    values: dict[str, object] = {}
    for key, value in dict(state.metrics).items():
        values[f"train/{key}"] = value
    # ``sampler_stats`` is a typed SamplerStats record that composes its own
    # metric names; status never re-spells or re-flattens them.
    sampler_stats = state.sampler_stats
    if sampler_stats is not None:
        for key, value in sampler_stats.as_metrics().items():
            values[f"train/sampler/{key}"] = value
    return values


def _format_status_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "ok" if value else "failed"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        abs_value = abs(value)
        if 0 < abs_value < 1.0e-3 or abs_value >= 1.0e4:
            return f"{value:.3e}"
        return f"{value:.6g}"
    return _quote_value(str(value)) if _needs_shell_quote(str(value)) else str(value)


def _format_gib(value: int | float) -> str:
    return f"{float(value) / (1024**3):.1f}GB"


def _format_status_box(
    title: str,
    rows: Sequence[tuple[str, object] | None],
    *,
    max_line_width: int = _STATUS_BOX_MAX_LINE_WIDTH,
) -> list[str]:
    rendered_rows: list[tuple[str, str] | None] = []
    for row in rows:
        if row is None:
            if rendered_rows and rendered_rows[-1] is not None:
                rendered_rows.append(None)
            continue
        label, value = row
        rendered_rows.append((str(label), _format_box_value(value)))
    if rendered_rows and rendered_rows[-1] is None:
        rendered_rows.pop()

    label_width = max((len(label) for row in rendered_rows if row is not None for label, _ in [row]), default=0)
    max_line_width = _validate_max_line_width(max_line_width)
    max_content_width = max_line_width - _STATUS_BOX_BORDER_WIDTH
    max_value_width = max(1, max_content_width - label_width - len(_STATUS_BOX_SEPARATOR))
    value_width = min(
        max((len(value) for row in rendered_rows if row is not None for _, value in [row]), default=0),
        max_value_width,
    )
    content_width = max(len(title), label_width + len(_STATUS_BOX_SEPARATOR) + value_width)
    top = "+" + "=" * (content_width + 2) + "+"
    rule = "+" + "-" * (content_width + 2) + "+"
    lines = [top, f"| {title.center(content_width)} |", rule]
    for row in rendered_rows:
        if row is None:
            lines.append(rule)
            continue
        label, value = row
        value_lines = _wrap_box_value(value, width=max_value_width)
        for index, value_line in enumerate(value_lines):
            rendered_label = label if index == 0 else ""
            text = f"{rendered_label.ljust(label_width)}{_STATUS_BOX_SEPARATOR}{value_line}"
            lines.append(f"| {text.ljust(content_width)} |")
    lines.append(top)
    return lines


def _wrap_box_value(value: str, *, width: int) -> list[str]:
    return textwrap.wrap(
        value,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    ) or ["null"]


def _format_box_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_status_value(value)
    return " ".join(str(value).splitlines()) or "null"


def _quote_value(value: str) -> str:
    return json.dumps(value)


def _needs_shell_quote(value: str) -> bool:
    return any(character.isspace() for character in value) or value == ""


def _validate_max_line_width(value: int) -> int:
    width = int(value)
    if width < 40:
        raise ValueError(f"max_line_width must be at least 40, got {width}")
    return width


__all__ = ["Status"]
