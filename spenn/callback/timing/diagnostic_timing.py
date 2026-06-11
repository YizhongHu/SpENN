"""Diagnostic timing callback."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, Callable

from .base import Callback, Event, _attach_event_metrics, _sync_cuda


class DiagnosticTiming(Callback):
    """Measure per-diagnostic evaluation durations."""

    def __init__(
        self,
        triggers: Iterable[str] = ("diagnostic_start", "diagnostic_end", "diagnostic_failed"),
        *,
        cuda_synchronize: bool = False,
        clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.cuda_synchronize = bool(cuda_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._starts: dict[tuple[int, str], float] = {}

    def on_diagnostic_start(self, event: Event) -> None:
        """Record one diagnostic start time."""

        key = self._event_key(event)
        _sync_cuda(self.cuda_synchronize)
        self._starts[key] = self.clock()

    def on_diagnostic_end(self, event: Event) -> None:
        """Log one diagnostic duration."""

        self._log_end(event, failed=False)

    def on_diagnostic_failed(self, event: Event) -> None:
        """Log one diagnostic failure duration when possible."""

        self._log_end(event, failed=True)

    def _log_end(self, event: Event, *, failed: bool) -> None:
        key = self._event_key(event)
        if key not in self._starts:
            return
        _sync_cuda(self.cuda_synchronize)
        duration = self.clock() - self._starts.pop(key)
        metrics: dict[str, float | bool] = {"time_sec": duration}
        if failed:
            metrics["failed"] = True
        step, diagnostic_name = key
        namespace = f"diagnostics/{diagnostic_name}"
        event.context.log(metrics, step=step, namespace=namespace)
        _attach_event_metrics(event, namespace, metrics)

    def _event_key(self, event: Event) -> tuple[int, str]:
        name = event.payload.get("diagnostic_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("diagnostic timing events require a non-empty diagnostic_name payload")
        step = 0 if event.step is None else int(event.step)
        return step, name


__all__ = ["DiagnosticTiming"]
