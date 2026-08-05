"""Training-step timing callback."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable
from typing import Any, Callable

from .base import Callback, Event, _attach_event_metrics, _sync_cuda


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


__all__ = ["TrainStepTiming"]
