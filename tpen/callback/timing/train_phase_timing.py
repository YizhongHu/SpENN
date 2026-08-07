"""Training-loop phase timing callback."""

from __future__ import annotations

import time
from typing import Any, Callable

from tpen.artifacts import RunContext
from tpen.events import Ended, Event as TypedEvent, Occurrence, Started
from tpen.events import Subscription, ended, started

from ..cadence import Cadence, SubscriptionGroup
from .base import Callback, _sync_cuda


class TrainPhaseTiming(Callback):
    """Measure typed training-phase durations within each training iteration.

    Every trainer phase is a typed `TrainingPhase` scope, so this callback is
    trigger-free: it observes ``Started``/``Ended`` boundaries instead of named
    legacy events. Each phase type owns the durable metric fragment
    ``phase_name``, and the measured key is ``f"{phase_name}_time_sec"``.
    Successful ``TrainingIterationCompleted`` events report one ``train/perf``
    record; unconditional ``Ended[TrainingIteration]`` observation clears all
    partial state, including failed or cadence-skipped iterations.
    Scheduling scalar options gate only typed completion reporting; phase
    collection and iteration cleanup remain ungated.

    Parameters
    ----------
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
        *,
        cuda_synchronize: bool = False,
        clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        # Importing ``tpen.callback.timing`` must stay torch-free. Resolve the
        # training-owned event types only when this callback is constructed.
        from tpen.training.events import (
            TrainingIteration,
            TrainingIterationCompleted,
            TrainingPhase,
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
            # TrainingIteration is not a TrainingPhase, so these selectors do
            # not overlap the gated completion group below.
            SubscriptionGroup(
                selectors=(
                    started(TrainingPhase),
                    ended(TrainingPhase),
                    ended(TrainingIteration),
                )
            ),
            SubscriptionGroup(
                selectors=(Subscription.of(TrainingIterationCompleted),),
                cadence=report_cadence,
            ),
        )
        super().__init__(typed_groups=typed_groups, **kwargs)
        self.cuda_synchronize = bool(cuda_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._phase_type = TrainingPhase
        self._training_iteration_type = TrainingIteration
        self._completion_type = TrainingIterationCompleted
        # Keyed by the paired scope coordinate ``(concrete phase type, count)``
        # so Started and Ended always match; the value carries the durable step.
        self._phase_starts: dict[tuple[type[object], int], tuple[int, float]] = {}
        self._durations: dict[int, dict[str, float]] = {}

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Measure, report, or clean up one admitted training occurrence."""

        event = occurrence.event
        if isinstance(event, Started) and isinstance(event.operation, self._phase_type):
            key = (type(event.operation), occurrence.count)
            _sync_cuda(self.cuda_synchronize)
            self._phase_starts[key] = (int(event.operation.step), self.clock())
            return
        if isinstance(event, Ended) and isinstance(event.operation, self._phase_type):
            key = (type(event.operation), occurrence.count)
            start_record = self._phase_starts.pop(key, None)
            if start_record is None:
                return
            _sync_cuda(self.cuda_synchronize)
            step, start = start_record
            # The metric fragment is owned by the phase type, never re-spelled
            # here, so timing keys cannot drift away from the phase contract.
            metric_key = f"{type(event.operation).phase_name}_time_sec"
            metrics = self._durations.setdefault(step, {})
            if metric_key in metrics:
                raise RuntimeError(f"duplicate {metric_key} for training step {step}")
            metrics[metric_key] = self.clock() - start
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

    def _report_completed_iteration(self, step: int, context: RunContext) -> None:
        """Log measurements for one admitted successful iteration."""

        metrics = self._durations.pop(step, None)
        if metrics:
            context.log(metrics, step=step, namespace="train/perf")

    def _cleanup_iteration(self, step: int) -> None:
        """Discard all measurements for an ended iteration without side effects."""

        self._phase_starts = {
            key: value for key, value in self._phase_starts.items() if value[0] != step
        }
        self._durations.pop(step, None)

    def _reset_typed_state(self) -> None:
        """Clear timing caches when the owning RunContext identity changes."""

        self._phase_starts.clear()
        self._durations.clear()


__all__ = ["TrainPhaseTiming"]
