"""Training-loop phase timing callback."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, Callable

from tpen.artifacts import RunContext
from tpen.events import Ended, Event as TypedEvent, Occurrence, Started
from tpen.events import Subscription, ended, started

from ..cadence import Cadence, SubscriptionGroup
from .base import Callback, Event, _sync_cuda


class TrainPhaseTiming(Callback):
    """Measure named training-loop phase durations within each step.

    Sample collection uses typed ``Started[CollectSamples]`` and
    ``Ended[CollectSamples]`` occurrences. The remaining trainer phases still
    use legacy ``train_phase_start``/``train_phase_end`` events. Successful
    ``TrainingIterationCompleted`` events report one ``train/perf`` record;
    unconditional ``Ended[TrainingIteration]`` observation clears all partial
    state, including failed or cadence-skipped iterations.
    Scheduling scalar options gate only typed completion reporting; legacy
    phase-boundary collection and iteration cleanup remain ungated.

    Parameters
    ----------
    triggers : iterable of str, optional
        Legacy phase-boundary event names that should trigger collection.
    every_n_steps : int or None, optional
        Interval between successful typed completion reports. ``None`` reports
        every successful completion occurrence.
    start_step : int, optional
        Legacy zero-based first report coordinate. It maps to the one-based
        typed occurrence start ``start_step + 1``.
    max_calls : int or None, optional
        Maximum admitted successful typed completion occurrences.
    probability : float, optional
        Admission probability for an otherwise eligible typed completion.
    seed : int or None, optional
        Seed for the typed reporting group's independent RNG stream.
    cuda_synchronize : bool, optional
        Synchronize CUDA at phase boundaries for accurate device timing.
    clock : callable, optional
        Monotonic clock override for deterministic tests.
    """

    def __init__(
        self,
        triggers: Iterable[str] = ("train_phase_start", "train_phase_end"),
        *,
        cuda_synchronize: bool = False,
        clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        # Importing ``tpen.callback.timing`` must stay torch-free. Resolve the
        # training-owned event types only when this callback is constructed.
        from tpen.training.events import (
            CollectSamples,
            TrainingIteration,
            TrainingIterationCompleted,
        )

        every_n_steps = kwargs.pop("every_n_steps", None)
        start_step = int(kwargs.pop("start_step", 0))
        max_calls = kwargs.pop("max_calls", None)
        probability = kwargs.pop("probability", 1.0)
        seed = kwargs.pop("seed", None)
        report_cadence = Cadence(
            every_n=1 if every_n_steps is None else int(every_n_steps),
            # Legacy step 0 is the first one-based occurrence coordinate.
            start=start_step + 1,
            max_calls=max_calls,
            probability=probability,
            seed=seed,
        )
        typed_groups = (
            SubscriptionGroup(
                selectors=(
                    started(CollectSamples),
                    ended(CollectSamples),
                    ended(TrainingIteration),
                )
            ),
            SubscriptionGroup(
                selectors=(Subscription.of(TrainingIterationCompleted),),
                cadence=report_cadence,
            ),
        )
        super().__init__(triggers, typed_groups=typed_groups, **kwargs)
        self.cuda_synchronize = bool(cuda_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._collect_samples_type = CollectSamples
        self._training_iteration_type = TrainingIteration
        self._completion_type = TrainingIterationCompleted
        self._starts: dict[tuple[int, str], float] = {}
        self._collect_samples_starts: dict[
            tuple[type[object], int], tuple[int, float]
        ] = {}
        self._durations: dict[int, dict[str, float]] = {}

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Measure, report, or clean up one admitted training occurrence."""

        event = occurrence.event
        if isinstance(event, Started) and isinstance(
            event.operation, self._collect_samples_type
        ):
            step = int(event.operation.step)
            key = (type(event.operation), occurrence.count)
            _sync_cuda(self.cuda_synchronize)
            self._collect_samples_starts[key] = (step, self.clock())
            return
        if isinstance(event, Ended) and isinstance(
            event.operation, self._collect_samples_type
        ):
            key = (type(event.operation), occurrence.count)
            start_record = self._collect_samples_starts.pop(key, None)
            if start_record is None:
                return
            _sync_cuda(self.cuda_synchronize)
            step, start = start_record
            metrics = self._durations.setdefault(step, {})
            if "sampling_time_sec" in metrics:
                raise RuntimeError(
                    f"duplicate sampling_time_sec for training step {step}"
                )
            metrics["sampling_time_sec"] = self.clock() - start
            return
        if isinstance(event, self._completion_type):
            self._report_completed_iteration(
                int(event.iteration.step),
                context,
            )
            return
        if isinstance(event, Ended) and isinstance(
            event.operation, self._training_iteration_type
        ):
            self._cleanup_iteration(int(event.operation.step))

    def on_train_phase_start(self, event: Event) -> None:
        """Record one phase start time."""

        key = self._event_key(event)
        _sync_cuda(self.cuda_synchronize)
        self._starts[key] = self.clock()

    def on_train_phase_end(self, event: Event) -> None:
        """Accumulate one phase duration for the enclosing step."""

        key = self._event_key(event)
        start = self._starts.pop(key, None)
        if start is None:
            return
        _sync_cuda(self.cuda_synchronize)
        step, phase = key
        self._durations.setdefault(step, {})[f"{phase}_time_sec"] = self.clock() - start

    def _report_completed_iteration(self, step: int, context: RunContext) -> None:
        """Log measurements for one admitted successful iteration."""

        metrics = self._durations.pop(step, None)
        if metrics:
            context.log(metrics, step=step, namespace="train/perf")

    def _cleanup_iteration(self, step: int) -> None:
        """Discard all measurements for an ended iteration without side effects."""

        self._starts = {key: value for key, value in self._starts.items() if key[0] != step}
        self._collect_samples_starts = {
            key: value
            for key, value in self._collect_samples_starts.items()
            if value[0] != step
        }
        self._durations.pop(step, None)

    def _reset_typed_state(self) -> None:
        """Clear timing caches when the owning RunContext identity changes."""

        self._starts.clear()
        self._collect_samples_starts.clear()
        self._durations.clear()

    @staticmethod
    def _event_key(event: Event) -> tuple[int, str]:
        """Return the ``(step, phase)`` identity of one phase event."""

        phase = event.payload.get("phase")
        if not phase:
            raise ValueError("train phase timing events require a 'phase' payload entry")
        step = event.step
        return (-1 if step is None else int(step), str(phase))


__all__ = ["TrainPhaseTiming"]
