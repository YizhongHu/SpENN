"""Callback primitives for configured SpENN runs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from spenn.artifacts import RunContext, write_json


@dataclass
class Event:
    """Lifecycle event delivered to callbacks."""

    name: str
    context: RunContext
    state: object | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def step(self) -> int | None:
        """Return the event step when available."""

        if "step" in self.payload:
            return None if self.payload["step"] is None else int(self.payload["step"])
        value = getattr(self.state, "global_step", None)
        return None if value is None else int(value)


class Callback:
    """Base class for event-triggered run callbacks.

    Parameters
    ----------
    triggers : iterable of str
        Event names that should trigger this callback.
    every_n_steps : int or None, optional
        Optional periodic step filter.
    start_step : int, optional
        First eligible step for periodic callbacks.
    max_calls : int or None, optional
        Maximum number of callback invocations.
    """

    def __init__(
        self,
        triggers: Iterable[str],
        every_n_steps: int | None = None,
        start_step: int = 0,
        max_calls: int | None = None,
    ) -> None:
        self.triggers = tuple(triggers)
        self.every_n_steps = every_n_steps
        self.start_step = int(start_step)
        self.max_calls = max_calls
        self.num_calls = 0

    def should_run(self, event: Event) -> bool:
        """Return whether this callback should handle `event`."""

        if event.name not in self.triggers:
            return False
        if self.max_calls is not None and self.num_calls >= self.max_calls:
            return False
        if self.every_n_steps is None:
            return True
        step = event.step
        if step is None or step < self.start_step:
            return False
        return (step - self.start_step) % self.every_n_steps == 0

    def handle(self, event: Event) -> None:
        """Handle an event if this callback is subscribed to it."""

        if not self.should_run(event):
            return
        method = getattr(self, f"on_{event.name}", None)
        if method is not None:
            method(event)
        self.num_calls += 1


class ConfigSnapshot(Callback):
    """Write a re-runnable run configuration at run start."""

    def __init__(self, triggers: Iterable[str], output_path: str | Path, **kwargs: Any) -> None:
        super().__init__(triggers, **kwargs)
        self.output_path = Path(output_path)

    def on_run_start(self, event: Event) -> None:
        """Write ``config.yaml``."""

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(event.context.source_cfg, self.output_path, resolve=False)


class ResolvedConfigSnapshot(Callback):
    """Write the fully resolved run configuration at run start."""

    def __init__(self, triggers: Iterable[str], output_path: str | Path, **kwargs: Any) -> None:
        super().__init__(triggers, **kwargs)
        self.output_path = Path(output_path)

    def on_run_start(self, event: Event) -> None:
        """Write ``resolved_config.yaml``."""

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(event.context.cfg, self.output_path, resolve=True)


class Metadata(Callback):
    """Write run metadata during lifecycle transitions."""

    def __init__(self, triggers: Iterable[str], output_path: str | Path, **kwargs: Any) -> None:
        super().__init__(triggers, **kwargs)
        self.output_path = Path(output_path)

    def on_run_start(self, event: Event) -> None:
        """Record running metadata."""

        self._write(event, status="running")

    def on_run_end(self, event: Event) -> None:
        """Record completed metadata."""

        self._write(event, status="completed")

    def on_exception(self, event: Event) -> None:
        """Record failed metadata."""

        self._write(event, status="failed")

    def _write(self, event: Event, *, status: str) -> None:
        metadata = event.context.metadata
        metadata.status = status
        data = metadata.to_dict()
        data["status"] = status
        exception = event.payload.get("exception")
        if exception is not None:
            data["exception_type"] = type(exception).__name__
            data["exception_message"] = str(exception)
        write_json(self.output_path, data)


class Status(Callback):
    """Write lifecycle status for one run."""

    def __init__(self, triggers: Iterable[str], output_path: str | Path, **kwargs: Any) -> None:
        super().__init__(triggers, **kwargs)
        self.output_path = Path(output_path)
        self.start_time: str | None = None

    def on_run_start(self, event: Event) -> None:
        """Record run start."""

        self.start_time = _now()
        self._write(
            status="running",
            current_event=event.name,
            end_time=None,
            exception_type=None,
            exception_message=None,
        )

    def on_run_end(self, event: Event) -> None:
        """Record successful completion."""

        self._write(
            status="completed",
            current_event=event.name,
            end_time=_now(),
            exception_type=None,
            exception_message=None,
        )

    def on_exception(self, event: Event) -> None:
        """Record run failure."""

        exception = event.payload.get("exception")
        self._write(
            status="failed",
            current_event=event.name,
            end_time=_now(),
            exception_type=None if exception is None else type(exception).__name__,
            exception_message=None if exception is None else str(exception),
        )

    def _write(
        self,
        *,
        status: str,
        current_event: str,
        end_time: str | None,
        exception_type: str | None,
        exception_message: str | None,
    ) -> None:
        write_json(
            self.output_path,
            {
                "status": status,
                "start_time": self.start_time,
                "end_time": end_time,
                "current_event": current_event,
                "exception_type": exception_type,
                "exception_message": exception_message,
            },
        )


class ReportSkeleton(Callback):
    """Write a human-readable scaffold report."""

    def __init__(self, triggers: Iterable[str], output_path: str | Path, **kwargs: Any) -> None:
        super().__init__(triggers, **kwargs)
        self.output_path = Path(output_path)

    def on_run_end(self, event: Event) -> None:
        """Write ``report.md`` for a scaffold run."""

        run_id = event.context.metadata.run_id
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            "\n".join(
                [
                    "# SpENN Scaffold Run",
                    "",
                    f"- Run ID: `{run_id}`",
                    f"- Run directory: `{event.context.run_dir}`",
                    "- Status: completed",
                    "",
                    "This scaffold run only exercised generic run management.",
                    "No Hooke physics, VMC training, diagnostics, sampling, or plotting were run.",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "Callback",
    "ConfigSnapshot",
    "Event",
    "Metadata",
    "ReportSkeleton",
    "ResolvedConfigSnapshot",
    "Status",
]
