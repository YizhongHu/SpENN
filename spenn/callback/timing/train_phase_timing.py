"""Training-loop phase timing callback."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, Callable

from .base import Callback, Event, _attach_event_metrics, _sync_cuda


class TrainPhaseTiming(Callback):
    """Measure named training-loop phase durations within each step.

    ``VMCTrainer.fit`` emits ``train_phase_start``/``train_phase_end`` around
    each loop phase (``sampling``, ``batch_build``, ``local_energy``,
    ``forward``, ``objective``, ``backward``, ``optimizer_step``,
    ``post_step_metrics``). Durations accumulate per step and are logged as a
    single ``train/perf`` record at ``step_end``, one ``<phase>_time_sec`` key
    per phase observed in that step.

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
        self._durations: dict[int, dict[str, float]] = {}

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
