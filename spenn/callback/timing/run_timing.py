"""Whole-run timing callback."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, Callable

from .base import Callback, Event, _attach_event_metrics, _sync_cuda


class RunTiming(Callback):
    """Measure whole-run timestamps and wall-clock duration."""

    def __init__(
        self,
        triggers: Iterable[str] = ("run_start", "run_end", "exception", "run_failed"),
        *,
        log_start_end_timestamps: bool = True,
        log_wall_time: bool = True,
        cuda_synchronize: bool = False,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.log_start_end_timestamps = bool(log_start_end_timestamps)
        self.log_wall_time = bool(log_wall_time)
        self.cuda_synchronize = bool(cuda_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self.wall_clock = time.time if wall_clock is None else wall_clock
        self._start_perf: float | None = None

    def on_run_start(self, event: Event) -> None:
        """Record run start timing."""

        _sync_cuda(self.cuda_synchronize)
        self._start_perf = self.clock()
        if self.log_start_end_timestamps:
            event.context.log({"start_time_unix": self.wall_clock()}, step=0, namespace="runtime")

    def on_run_end(self, event: Event) -> None:
        """Log whole-run elapsed time at normal completion."""

        self._log_end(event, failed=False)

    def on_exception(self, event: Event) -> None:
        """Log whole-run elapsed time on failure without swallowing the exception."""

        self._log_end(event, failed=True)

    def on_run_failed(self, event: Event) -> None:
        """Alias for runtimes that emit ``run_failed``."""

        self._log_end(event, failed=True)

    def _log_end(self, event: Event, *, failed: bool) -> None:
        _sync_cuda(self.cuda_synchronize)
        now = self.clock()
        metrics: dict[str, float | bool] = {}
        if self.log_start_end_timestamps:
            metrics["end_time_unix"] = self.wall_clock()
        if self.log_wall_time and self._start_perf is not None:
            metrics["wall_time_sec"] = now - self._start_perf
        if failed:
            metrics["failed"] = True
        if metrics:
            event.context.log(metrics, step=0, namespace="runtime")
            _attach_event_metrics(event, "runtime", metrics)


__all__ = ["RunTiming"]
