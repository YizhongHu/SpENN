"""Terminal and artifact status callbacks."""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from spenn.artifacts import write_json

from .base import Callback, Event

_STATUS_BOX_MAX_LINE_WIDTH = 100
_STATUS_BOX_BORDER_WIDTH = 4
_STATUS_BOX_SEPARATOR = " : "


class Status(Callback):
    """Write lifecycle status artifacts and terminal status lines."""

    def __init__(
        self,
        triggers: Iterable[str],
        output_path: str | Path | None = None,
        *,
        terminal: bool = True,
        logger_name: str = "spenn.status",
        include: Sequence[str] | None = None,
        color: str = "auto",
        max_line_width: int = _STATUS_BOX_MAX_LINE_WIDTH,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.output_path = None if output_path is None else Path(output_path)
        self.terminal = bool(terminal)
        self.logger = logging.getLogger(logger_name)
        self.include = tuple(_DEFAULT_STATUS_METRICS if include is None else include)
        self.color = _validate_terminal_choice(color, name="color")
        self.max_line_width = _validate_max_line_width(max_line_width)
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

    def on_step_end(self, event: Event) -> None:
        """Write one compact training status line."""

        line = _format_train_status(event, self.include)
        if line is not None:
            self._log_status(line, kind="train")

    def on_evaluate_end(self, event: Event) -> None:
        """Write one compact evaluation status line."""

        line = _format_evaluate_status(event)
        if line is not None:
            self._log_status(line, kind="eval")

    def _log_status(self, line: str, *, kind: str) -> None:
        if not self.terminal:
            return
        self.logger.info(_color_status_line(line, kind=kind, color=self.color))

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


def configure_terminal_logging(
    *,
    enabled: bool = True,
    level: str = "info",
    color: str = "auto",
    logger_name: str = "spenn",
) -> None:
    """Configure the package terminal logging channel.

    Parameters
    ----------
    enabled : bool, optional
        If ``False``, leave logging configuration unchanged.
    level : str, optional
        Logging level name.
    color : {"auto", "always", "never"}, optional
        Accepted for config validation and consistency with `Status`.
    logger_name : str, optional
        Logger subtree to configure.
    """

    if not enabled:
        return
    _validate_terminal_choice(color, name="color")
    logger = logging.getLogger(logger_name)
    logger.setLevel(_logging_level(level))
    for handler in logger.handlers:
        if getattr(handler, "_spenn_terminal_handler", False):
            handler.setLevel(_logging_level(level))
            return
    handler = logging.StreamHandler()
    handler._spenn_terminal_handler = True
    handler.setLevel(_logging_level(level))
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def _now(event: Event) -> str:
    return event.context.now_iso()


_DEFAULT_STATUS_METRICS = (
    "train/loss",
    "train/energy",
    "train/energy_stderr",
    "train/sampler/acceptance_rate",
    "train/grad_norm",
    "train/local_energy_finite_fraction",
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

_STATUS_COLORS = {
    "run": "\033[36m",
    "train": "\033[34m",
    "eval": "\033[35m",
    "completed": "\033[32m",
    "failed": "\033[31m",
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
        *_format_status_box("SpENN Run Status", status_rows, max_line_width=max_line_width),
        *_format_status_box("Hardware Environment", hardware_rows, max_line_width=max_line_width),
    ]


def _format_run_end(event: Event) -> str:
    return f"[run] completed dir={event.context.metadata.run_dir}"


def _format_run_failure(event: Event, exception: object | None) -> str:
    parts = ["[run] failed", f"dir={event.context.metadata.run_dir}"]
    if exception is not None:
        parts.extend([f"exception={type(exception).__name__}", f"message={_quote_value(str(exception))}"])
    return " ".join(parts)


def _format_train_status(event: Event, include: Sequence[str]) -> str | None:
    state = event.state
    if state is None:
        return None
    values = _training_metric_values(state)
    values.update(_payload_metric_values(event))
    rendered = [
        f"{_STATUS_LABELS.get(identity, identity)}={_format_status_value(values[identity])}"
        for identity in include
        if identity in values
    ]
    if not rendered:
        return None
    step = event.step
    prefix = "[train]" if step is None else f"[train] step={step}"
    return " ".join([prefix, *rendered])


def _format_evaluate_status(event: Event) -> str | None:
    metrics = event.payload.get("metrics")
    values = {}
    if isinstance(metrics, Mapping):
        values.update({f"eval/{key}": value for key, value in metrics.items()})
    values.update(_payload_metric_values(event))
    if not values:
        return None
    include = ("eval/energy", "eval/energy_stderr", "eval/energy_error", "eval/perf/wall_time_sec")
    labels = {
        "eval/energy": "energy",
        "eval/energy_stderr": "stderr",
        "eval/energy_error": "abs_error",
        "eval/perf/wall_time_sec": "wall_time",
    }
    rendered = [
        f"{labels[identity]}={_format_status_value(values[identity])}"
        for identity in include
        if identity in values
    ]
    if not rendered:
        return None
    return " ".join(["[eval]", *rendered])


def _training_metric_values(state: object) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in dict(getattr(state, "metrics", {}) or {}).items():
        values[f"train/{key}"] = value
    for key, value in dict(getattr(state, "sampler_stats", {}) or {}).items():
        values[f"train/sampler/{key}"] = value
    return values


def _payload_metric_values(event: Event) -> dict[str, object]:
    values: dict[str, object] = {}
    by_namespace = event.payload.get("metrics_by_namespace")
    if not isinstance(by_namespace, Mapping):
        return values
    for namespace, metrics in by_namespace.items():
        if not isinstance(namespace, str) or not isinstance(metrics, Mapping):
            continue
        for key, value in metrics.items():
            values[f"{namespace}/{key}"] = value
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


def _validate_terminal_choice(value: str, *, name: str) -> str:
    if value not in {"auto", "always", "never"}:
        raise ValueError(f"{name} must be one of 'auto', 'always', or 'never', got {value!r}")
    return value


def _validate_max_line_width(value: int) -> int:
    width = int(value)
    if width < 40:
        raise ValueError(f"max_line_width must be at least 40, got {width}")
    return width


def _logging_level(level: str) -> int:
    value = getattr(logging, str(level).upper(), None)
    if not isinstance(value, int):
        raise ValueError(f"Unsupported logging level {level!r}")
    return value


def _color_status_line(line: str, *, kind: str, color: str) -> str:
    if not _color_enabled(color):
        return line
    prefix = _STATUS_COLORS.get(kind)
    if prefix is None:
        return line
    return f"{prefix}{line}\033[0m"


def _color_enabled(color: str) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if color == "always":
        return True
    if color == "never":
        return False
    if os.environ.get("SLURM_JOB_ID"):
        return False
    return sys.stderr.isatty()



__all__ = ["Status", "configure_terminal_logging"]
