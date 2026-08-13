"""Typed training-iteration timing callback."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable

from tpen.artifacts import RunContext
from tpen.events import (
    Ended,
    Event as TypedEvent,
    Occurrence,
    Started,
    TrainingTiming,
    TrainingTimingState,
    ended,
    started,
)

from ..base import StatefulCallback
from ..cadence import SubscriptionGroup
from .base import _sync_device


class TrainStepTiming(StatefulCallback[TrainingTimingState]):
    """Measure completed ``TrainingIteration`` scopes."""

    state_type = TrainingTimingState

    def __init__(
        self,
        *,
        rolling_window: int = 20,
        accelerator_synchronize: bool = False,
        clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        from tpen.training.events import TrainingIteration

        super().__init__(
            typed_groups=(
                SubscriptionGroup(selectors=(started(TrainingIteration), ended(TrainingIteration))),
            ),
            **kwargs,
        )
        if rolling_window <= 0:
            raise ValueError(f"rolling_window must be positive, got {rolling_window}")
        self.rolling_window = int(rolling_window)
        self.accelerator_synchronize = bool(accelerator_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._iteration_type = TrainingIteration
        self._starts: dict[tuple[type[object], int], tuple[int, float]] = {}
        self._durations: deque[float] = deque(maxlen=self.rolling_window)

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext, state: TrainingTimingState
    ) -> None:
        event = occurrence.event
        if isinstance(event, Started) and isinstance(event.operation, self._iteration_type):
            _sync_device(self.accelerator_synchronize)
            self._starts[(type(event.operation), occurrence.count)] = (
                event.operation.step,
                self.clock(),
            )
            return
        if not isinstance(event, Ended) or not isinstance(event.operation, self._iteration_type):
            return
        record = self._starts.pop((type(event.operation), occurrence.count), None)
        if record is None:
            return
        _sync_device(self.accelerator_synchronize)
        step, start = record
        duration = self.clock() - start
        self._durations.append(duration)
        metrics = {
            "step_time_sec": duration,
            "step_time_sec_rolling_mean": sum(self._durations) / len(self._durations),
        }
        state.timing = TrainingTiming(**metrics)
        context.log(metrics, step=step, namespace="train/perf")

    def _reset_typed_state(self) -> None:
        self._starts.clear()


__all__ = ["TrainStepTiming"]
