"""Evaluation timing callback."""

from __future__ import annotations

import time
from typing import Any, Callable

from tpen.artifacts import RunContext
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.run_events import RunFailed

from ..cadence import SubscriptionGroup
from .base import Callback, TimingSource, _occurrence_time, _sync_device


class EvaluationTiming(Callback):
    """Measure evaluation wall time.

    Completely data-free: it reads nothing from any event, any payload, and any
    state, only the moment each boundary happened. That is why it is a plain
    `tpen.callback.Callback` rather than a `tpen.callback.StatefulCallback`, and
    it is now fully trigger-free.

    Notes
    -----
    This callback observes TWO domains' moments, which is what made it the last
    holder of a legacy run-level trigger. Two boundaries belong to the
    evaluation suite; the third, `tpen.run_events.RunFailed`, belongs to the run.
    The suite completion now carries its aggregate status, so both a returned
    failed suite and a raising run produce ``eval/perf {failed: True}`` without
    manufacturing a second lifecycle moment.

    Being a plain `Callback` is exactly why the migration works here and not on
    `tpen.callback.Status`: `tpen.artifacts.RunContext._dispatch_occurrence`
    skips a `tpen.callback.StatefulCallback` at every boundary carrying no state,
    and the run lifecycle carries none.

    All three selectors sit in ONE group. They share a single ungated decision,
    and `tpen.callback.cadence.validate_subscription_groups` rejects overlapping
    deliveries across groups.

    Parameters
    ----------
    accelerator_synchronize : bool, optional
        Synchronize the accelerator at both boundaries for device timing.
    clock : callable, optional
        Monotonic clock override for deterministic tests.
    """

    def __init__(
        self,
        *,
        accelerator_synchronize: bool = False,
        clock: Callable[[], float] | None = None,
        timing_backend: Any | None = None,
        device_backend: Any | None = None,
        **kwargs: Any,
    ) -> None:
        # Importing ``tpen.callback.timing`` must stay torch-free, and importing
        # anything from `tpen.evaluation` runs that package's ``__init__``, which
        # pulls in torch. Resolve the evaluation-owned event types only when this
        # callback is constructed -- the same reason `TrainPhaseTiming` defers
        # its training imports.
        from tpen.evaluation.events import EvaluationCompleted, EvaluationStarted

        super().__init__(
            typed_groups=(
                SubscriptionGroup(
                    selectors=(
                        Subscription.of(EvaluationStarted),
                        Subscription.of(EvaluationCompleted),
                        Subscription.of(RunFailed),
                    )
                ),
            ),
            **kwargs,
        )
        self.accelerator_synchronize = bool(accelerator_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._timing = TimingSource(clock=self.clock, backend=timing_backend, device_backend=device_backend)
        self._started_type = EvaluationStarted
        self._completed_type = EvaluationCompleted
        self._start: tuple[Any, Any | None] | None = None

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Start the clock at the suite's start and report at either outcome."""

        event = occurrence.event
        if isinstance(event, self._started_type):
            self._start_timing(occurrence)
            return
        if isinstance(event, self._completed_type):
            self._log_end(occurrence, context, failed=event.status == "failed")
            return
        # A run that failed before or without evaluating never started the clock,
        # and `_log_end` returns early for it, so this reports only a suite that
        # was genuinely in flight.
        if isinstance(event, RunFailed):
            self._log_end(occurrence, context, failed=True)

    def _start_timing(self, occurrence: Occurrence[TypedEvent]) -> None:
        _sync_device(self.accelerator_synchronize)
        self._start = self._timing.start(_occurrence_time(occurrence, self.clock))

    def _log_end(
        self, occurrence: Occurrence[TypedEvent], context: RunContext, *, failed: bool
    ) -> None:
        if self._start is None:
            return
        _sync_device(self.accelerator_synchronize)
        elapsed = self._timing.elapsed(self._start, _occurrence_time(occurrence, self.clock))
        metrics: dict[str, float | bool] = {"wall_time_sec": elapsed.host}
        if elapsed.device is not None:
            metrics["device_wall_time_sec"] = elapsed.device
        if failed:
            metrics["failed"] = True
        # Evaluation has no step coordinate: its coordinate is a task namespace
        # string, and every evaluation record has always been logged at step 0.
        # The 0 is written here rather than read from anywhere -- never from a
        # state cursor, whose value fields are stale above their assignment.
        context.log(metrics, step=0, namespace="eval/perf")
        self._start = None

    def _reset_typed_state(self) -> None:
        """Drop a half-open measurement when the owning RunContext changes."""

        self._start = None


__all__ = ["EvaluationTiming"]
