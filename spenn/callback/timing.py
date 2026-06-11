"""Runtime timing callbacks."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable
from typing import Any, Callable

import torch

from .base import Callback, Event, _attach_event_metrics


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


class TrainStepTiming(Callback):
    """Measure training step durations."""

    def __init__(
        self,
        triggers: Iterable[str] = ("step_start", "step_end"),
        *,
        rolling_window: int = 20,
        cuda_synchronize: bool = False,
        clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        if rolling_window <= 0:
            raise ValueError(f"rolling_window must be positive, got {rolling_window}")
        self.rolling_window = int(rolling_window)
        self.cuda_synchronize = bool(cuda_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._starts: dict[int, float] = {}
        self._durations: deque[float] = deque(maxlen=self.rolling_window)

    def on_step_start(self, event: Event) -> None:
        """Record the start time for one training step."""

        step = event.step
        if step is None:
            return
        _sync_cuda(self.cuda_synchronize)
        self._starts[int(step)] = self.clock()

    def on_step_end(self, event: Event) -> None:
        """Log step duration and rolling mean."""

        step = event.step
        if step is None or int(step) not in self._starts:
            return
        _sync_cuda(self.cuda_synchronize)
        duration = self.clock() - self._starts.pop(int(step))
        self._durations.append(duration)
        metrics = {
            "step_time_sec": duration,
            "step_time_sec_rolling_mean": sum(self._durations) / len(self._durations),
        }
        event.context.log(metrics, step=int(step), namespace="train/perf")
        _attach_event_metrics(event, "train/perf", metrics)


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


def _sync_cuda(cuda_synchronize: bool) -> None:
    if cuda_synchronize and torch.cuda.is_available():
        torch.cuda.synchronize()



__all__ = ["DiagnosticTiming", "EvaluationTiming", "RunTiming", "TrainStepTiming"]
