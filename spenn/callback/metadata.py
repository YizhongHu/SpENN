"""Run metadata lifecycle callbacks."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from spenn.artifacts import write_json

from .base import Callback, Event


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



__all__ = ["Metadata"]
