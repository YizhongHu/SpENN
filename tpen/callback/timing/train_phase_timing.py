"""Training-loop phase timing callback."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, Callable

from tpen.artifacts import RunContext
from tpen.events import Ended, Event as TypedEvent, Occurrence, Started

from .base import Callback, Event, _attach_event_metrics, _sync_cuda


class TrainPhaseTiming(Callback):
    """Measure named training-loop phase durations within each step.

    Sample collection uses typed ``Started[CollectSamples]`` and
    ``Ended[CollectSamples]`` occurrences. The remaining trainer phases still
    use the legacy ``train_phase_start``/``train_phase_end`` events during the
    incremental migration. Durations accumulate per step and are logged as a
    single ``train/perf`` record at ``step_end``. Reporting cadence remains
    controlled by the legacy ``step_end`` trigger until A2.

    Parameters
    ----------
    triggers : iterable of str, optional
        Event names that should trigger this callback.
    cuda_synchronize : bool, optional
        Synchronize CUDA at phase boundaries for accurate device timing.
    clock : callable, optional
        Monotonic clock override for deterministic tests.
    """

    def __init__(
        self,
        triggers: Iterable[str] = ("train_phase_start", "train_phase_end", "step_end"),
        *,
        cuda_synchronize: bool = False,
        clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.cuda_synchronize = bool(cuda_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._starts: dict[tuple[int, str], float] = {}
        self._collect_samples_starts: dict[int, tuple[int, float]] = {}
        self._durations: dict[int, dict[str, float]] = {}

    def handle_occurrence(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Measure the typed sample-collection pilot operation."""

        del context
        event = occurrence.event
        if not isinstance(event, (Started, Ended)):
            return
        # Keep callback import from loading the full training graph before a
        # typed training occurrence is actually dispatched.
        from tpen.training.events import CollectSamples

        operation = event.operation
        if not isinstance(operation, CollectSamples):
            return
        step = int(operation.step)
        if isinstance(event, Started):
            self._prune_typed_cache(step)
            _sync_cuda(self.cuda_synchronize)
            self._collect_samples_starts[occurrence.count] = (step, self.clock())
            return

        start_record = self._collect_samples_starts.pop(occurrence.count, None)
        if start_record is None:
            return
        _sync_cuda(self.cuda_synchronize)
        step, start = start_record
        metrics = self._durations.setdefault(step, {})
        if "sampling_time_sec" in metrics:
            raise RuntimeError(f"duplicate sampling_time_sec for training step {step}")
        metrics["sampling_time_sec"] = self.clock() - start

    def _prune_typed_cache(self, step: int) -> None:
        """Discard typed timing state that a skipped legacy trigger cannot report."""

        # CollectSamples is a sequential trainer phase in A1, so any unmatched
        # prior start is stale, including a failed retry of the same step.
        self._collect_samples_starts.clear()
        for cached_step in tuple(self._durations):
            if cached_step == step:
                continue
            metrics = self._durations[cached_step]
            metrics.pop("sampling_time_sec", None)
            if not metrics:
                del self._durations[cached_step]

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

    def on_step_end(self, event: Event) -> None:
        """Log all phase durations recorded for the finished step."""

        step = event.step
        if step is None:
            return
        step = int(step)
        # Unmatched starts from an aborted phase must not leak across steps.
        self._starts = {key: value for key, value in self._starts.items() if key[0] != step}
        self._collect_samples_starts = {
            count: value
            for count, value in self._collect_samples_starts.items()
            if value[0] != step
        }
        metrics = self._durations.pop(step, None)
        if not metrics:
            return
        event.context.log(metrics, step=step, namespace="train/perf")
        _attach_event_metrics(event, "train/perf", metrics)

    @staticmethod
    def _event_key(event: Event) -> tuple[int, str]:
        """Return the ``(step, phase)`` identity of one phase event."""

        phase = event.payload.get("phase")
        if not phase:
            raise ValueError("train phase timing events require a 'phase' payload entry")
        step = event.step
        return (-1 if step is None else int(step), str(phase))


__all__ = ["TrainPhaseTiming"]
