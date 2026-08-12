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
from tpen.run_events import RunCompleted, RunFailed, RunStarted
from tpen.training.events import TrainingIterationCompleted
from tpen.training.state import TrainerState

from .base import StatefulCallback
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
        Width the run-start status boxes are wrapped to.
    train_lines : bool, optional
        Whether to render one compact line per completed training iteration.
        Off by default, so a `Status` configured only for the run lifecycle does
        not start narrating the loop.
    **kwargs
        Forwarded to `StatefulCallback` (e.g. ``every_n_steps``).

    Notes
    -----
    This callback is now fully migrated, and it observes TWO different kinds of
    moment. The per-iteration training line reads
    `tpen.training.state.TrainerState` and arrives at
    `tpen.training.events.TrainingIterationCompleted`. The run lifecycle carries
    no domain state and arrives through a group declaring
    `tpen.callback.cadence.SubscriptionGroup.stateless`, which is the capability
    item ``24f91145`` added and this callback is the first consumer of. Before
    it, `tpen.artifacts.RunContext._dispatch_occurrence` decided delivery once
    per CALLBACK on ``isinstance(state, callback.state_type)``, so a run-level
    occurrence was SKIPPED here silently and ``status.json`` would have stopped
    being written with no error anywhere.

    WHAT THE THREE LEGACY STRINGS ACTUALLY FIRED ON, measured rather than
    assumed, because a wrong mapping here would drop a durable artifact in
    silence:

    - ``run_start`` was emitted by `tpen.run.run_from_config` (line 191) AND by
      each runner at the top of ``run()``, the second suppressed by the
      ``_run_start_emitted`` flag on both emitters. `tpen.run_events.RunStarted`
      is emitted once, by the harness only, one line earlier.
    - ``run_end`` was emitted by the RUNNERS, as the last statement of
      ``Train.run`` and ``Evaluate.run`` before ``return RunResult(...)`` --
      NOT from a ``finally``, so it never fired on a failure. Its typed
      counterpart `tpen.run_events.RunCompleted` is emitted by the harness
      immediately after ``runner.run`` returns, with nothing between the two
      points. Success path both ways; a suite whose tasks all failed still
      returns a result and still reaches it, exactly as before.
    - ``exception`` was emitted by the harness on the failure path only, on the
      line after ``run_failed``, both carrying one payload no consumer
      distinguished. `tpen.run_events.RunFailed` replaces both, is emitted eight
      lines earlier under the identical ``context is not None`` guard and the
      identical failure-swallowing helper, and carries the two strings this
      callback derived from the payload's live exception.

    Because ``run_end`` is success-only, `RunCompleted` alone is its faithful
    replacement and no failure case is dropped; the ``failed`` status keeps
    coming from the failure boundary, which is now `RunFailed`.

    ``current_event`` in ``status.json`` keeps the legacy STRINGS as its values.
    That field names a moment in a durable artifact rather than selecting an
    event, and rewriting shipped runs' vocabulary is not something a delivery
    migration gets to do (ADR-E006).

    The subscriptions are hardcoded rather than configured because ADR-E002
    forbids a config from naming the events a callback answers to.
    ``train_lines`` replaces the former string selector for step completion
    the training line, as a semantic option rather than an event selector.

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
            # An instance rendering no training line subscribes to no TRAINING
            # group, rather than subscribing and discarding every delivery. The
            # run-lifecycle group below is unconditional either way, because
            # ``status.json`` is written on every run.
            typed_groups=(
                (
                    SubscriptionGroup(
                        selectors=(Subscription.of(TrainingIterationCompleted),)
                    ),
                )
                if train_lines
                else ()
            )
            + (
                # One group with three selectors rather than three groups, the
                # shape `tpen.callback.timing.RunTiming` already uses for the
                # same three moments: they share one decision (observe the run
                # lifecycle) and no cadence gates any of them.
                SubscriptionGroup(
                    selectors=(
                        Subscription.of(RunStarted),
                        Subscription.of(RunCompleted),
                        Subscription.of(RunFailed),
                    ),
                    stateless=True,
                ),
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
        # The scheduling scalars gate the typed training line only. They could
        # never have gated the run-level boundaries on the legacy path either:
        # those carry no step, and `_CallbackCore.should_run` rejected a `None`
        # coordinate outright the moment `every_n_steps` was set.
        self._steps = StepCadenceGate(cadence)
        self.start_time: str | None = None

    def handle_stateless_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: Any
    ) -> None:
        """Write the status artifact and terminal line for one run boundary."""

        event = occurrence.event
        if isinstance(event, RunStarted):
            self._record_run_start(context)
        elif isinstance(event, RunCompleted):
            self._record_run_completed(context)
        elif isinstance(event, RunFailed):
            self._record_run_failed(context, event)

    def _record_run_start(self, context: RunContext) -> None:
        """Record run start."""

        self.start_time = context.now_iso()
        for line in _format_run_start_lines(context, max_line_width=self.max_line_width):
            self._log_status(line, kind="run")
        self._write(
            context,
            status="running",
            # The durable artifact's own vocabulary, not an event selector. See
            # the note on `Status`.
            current_event="run_start",
            end_time=None,
            exception_type=None,
            exception_message=None,
        )

    def _record_run_completed(self, context: RunContext) -> None:
        """Record successful completion."""

        self._log_status(_format_run_end(context), kind="completed")
        self._write(
            context,
            status="completed",
            current_event="run_end",
            end_time=context.now_iso(),
            exception_type=None,
            exception_message=None,
        )

    def _record_run_failed(self, context: RunContext, event: RunFailed) -> None:
        """Record run failure.

        The two strings are read off the typed event rather than derived from a
        live exception in an untyped payload. They are the same two values:
        `tpen.run.run_from_config` built the legacy payload's ``exception_type``
        and ``exception_message`` with ``type(exc).__name__`` and ``str(exc)``,
        which is exactly what this callback used to recompute here. The old
        ``exception is None`` branch went with them: the payload's ``exception``
        key was set unconditionally by the only emitter, so the branch never
        rendered, and `RunFailed` cannot carry an absent failure at all.
        """

        self._log_status(
            _format_run_failure(
                context,
                exception_type=event.exception_type,
                exception_message=event.exception_message,
            ),
            kind="failed",
        )
        self._write(
            context,
            status="failed",
            current_event="exception",
            end_time=context.now_iso(),
            exception_type=event.exception_type,
            exception_message=event.exception_message,
        )

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: Any,
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
        context: RunContext,
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
                "timezone": context.metadata.timezone,
                "start_time": self.start_time,
                "end_time": end_time,
                "current_event": current_event,
                "exception_type": exception_type,
                "exception_message": exception_message,
            },
        )


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

def _format_run_start_lines(context: RunContext, *, max_line_width: int = _STATUS_BOX_MAX_LINE_WIDTH) -> list[str]:
    metadata = context.metadata
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


def _format_run_end(context: RunContext) -> str:
    return f"[run] completed dir={context.metadata.run_dir}"


def _format_run_failure(context: RunContext, *, exception_type: str, exception_message: str) -> str:
    return " ".join(
        [
            "[run] failed",
            f"dir={context.metadata.run_dir}",
            f"exception={exception_type}",
            f"message={_quote_value(exception_message)}",
        ]
    )


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
