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
from ..cadence import Cadence, SubscriptionGroup
from .base import _occurrence_time, _sync_device


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

        every_n_steps = kwargs.pop("every_n_steps", None)
        start_step = int(kwargs.pop("start_step", 0))
        max_calls = kwargs.pop("max_calls", None)
        probability = kwargs.pop("probability", 1.0)
        seed = kwargs.pop("seed", None)
        cadence = Cadence(
            every_n=1 if every_n_steps is None else int(every_n_steps),
            start=start_step + 1,
            max_calls=max_calls,
            probability=probability,
            seed=seed,
        )
        super().__init__(
            typed_groups=(
                SubscriptionGroup(
                    selectors=(started(TrainingIteration), ended(TrainingIteration)),
                    cadence=cadence,
                ),
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
                _occurrence_time(occurrence, self.clock),
            )
            return
        if not isinstance(event, Ended) or not isinstance(event.operation, self._iteration_type):
            return
        record = self._starts.pop((type(event.operation), occurrence.count), None)
        if record is None:
            return
        _sync_device(self.accelerator_synchronize)
        step, start = record
        if not event.succeeded:
            return
        duration = _occurrence_time(occurrence, self.clock) - start
        self._durations.append(duration)
        metrics = {
            "step_time_sec": duration,
            "step_time_sec_rolling_mean": sum(self._durations) / len(self._durations),
        }
        state.timing = TrainingTiming(**metrics)
        context.log(metrics, step=step, namespace="train/perf")

    def _reset_typed_state(self) -> None:
        """Clear timing history when the owning run context changes."""

        self._starts.clear()
        self._durations.clear()

__all__ = ["TrainStepTiming"]
