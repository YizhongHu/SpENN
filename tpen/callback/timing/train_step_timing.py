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
from ..cadence import StepCadenceGate, SubscriptionGroup, pop_step_cadence
from .base import TimingSource, _occurrence_time, _sync_device


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

        cadence = pop_step_cadence(kwargs)
        timing_backend = kwargs.pop("timing_backend", None)
        device_backend = kwargs.pop("device_backend", None)
        super().__init__(
            typed_groups=(
                SubscriptionGroup(
                    selectors=(started(TrainingIteration), ended(TrainingIteration)),
                ),
            ),
            **kwargs,
        )
        if rolling_window <= 0:
            raise ValueError(f"rolling_window must be positive, got {rolling_window}")
        self.rolling_window = int(rolling_window)
        self.accelerator_synchronize = bool(accelerator_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._timing = TimingSource(clock=self.clock, backend=timing_backend, device_backend=device_backend)
        self._cadence_gate = StepCadenceGate(cadence)
        self._iteration_type = TrainingIteration
        self._starts: dict[tuple[type[object], int], tuple[int, tuple[Any, Any | None]]] = {}
        self._durations: deque[float] = deque(maxlen=self.rolling_window)

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext, state: TrainingTimingState
    ) -> None:
        event = occurrence.event
        if isinstance(event, Started) and isinstance(event.operation, self._iteration_type):
            if self.accelerator_synchronize:
                _sync_device(True)
            self._starts[(type(event.operation), occurrence.count)] = (
                event.operation.step,
                self._timing.start(_occurrence_time(occurrence, self.clock)),
            )
            return
        if not isinstance(event, Ended) or not isinstance(event.operation, self._iteration_type):
            return
        record = self._starts.pop((type(event.operation), occurrence.count), None)
        if record is None:
            return
        step, start = record
        # Consume the ended boundary before cadence admission. Runtime
        # occurrences already carry this emitter stamp; the fallback clock
        # must observe the same boundary sequence even when a sample is
        # intentionally skipped by the durable-step window.
        end_timestamp = _occurrence_time(occurrence, self.clock)
        if not event.succeeded or not self._cadence_gate.should_run(int(step)):
            return
        if self.accelerator_synchronize:
            _sync_device(True)
        elapsed = self._timing.elapsed(start, end_timestamp)
        duration = elapsed.host
        self._durations.append(duration)
        metrics = {
            "step_time_sec": duration,
            "step_time_sec_rolling_mean": sum(self._durations) / len(self._durations),
        }
        if elapsed.device is not None:
            metrics["step_device_time_sec"] = elapsed.device
        state.timing = TrainingTiming(**metrics)
        context.log(metrics, step=step, namespace="train/perf")

    def _reset_typed_state(self) -> None:
        """Clear timing history when the owning run context changes."""

        self._starts.clear()
        self._durations.clear()

__all__ = ["TrainStepTiming"]
