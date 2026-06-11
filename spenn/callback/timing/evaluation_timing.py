"""Evaluation timing callback."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, Callable

from .base import Callback, Event, _attach_event_metrics, _sync_cuda


class EvaluationTiming(Callback):
    """Measure evaluation wall time."""

    def __init__(
        self,
        triggers: Iterable[str] = ("evaluate_start", "evaluate_end", "eval_start", "eval_end", "exception", "eval_failed"),
        *,
        cuda_synchronize: bool = False,
        clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.cuda_synchronize = bool(cuda_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._start: float | None = None

    def on_evaluate_start(self, event: Event) -> None:
        """Record evaluation start."""

        self._start_timing()

    def on_eval_start(self, event: Event) -> None:
        """Alias for ``evaluate_start``."""

        self._start_timing()

    def on_evaluate_end(self, event: Event) -> None:
        """Log evaluation duration."""

        self._log_end(event, failed=False)

    def on_eval_end(self, event: Event) -> None:
        """Alias for ``evaluate_end``."""

        self._log_end(event, failed=False)

    def on_exception(self, event: Event) -> None:
        """Log elapsed evaluation time on failure when evaluation had started."""

        self._log_end(event, failed=True)

    def on_eval_failed(self, event: Event) -> None:
        """Alias for failed evaluation events."""

        self._log_end(event, failed=True)

    def _start_timing(self) -> None:
        _sync_cuda(self.cuda_synchronize)
        self._start = self.clock()

    def _log_end(self, event: Event, *, failed: bool) -> None:
        if self._start is None:
            return
        _sync_cuda(self.cuda_synchronize)
        metrics: dict[str, float | bool] = {"wall_time_sec": self.clock() - self._start}
        if failed:
            metrics["failed"] = True
        step = 0 if event.step is None else int(event.step)
        event.context.log(metrics, step=step, namespace="eval/perf")
        _attach_event_metrics(event, "eval/perf", metrics)
        self._start = None


__all__ = ["EvaluationTiming"]
