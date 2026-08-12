"""Evaluation timing callback."""

from __future__ import annotations

import time
from typing import Any, Callable

from tpen.artifacts import RunContext
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription

from ..cadence import SubscriptionGroup
from .base import Callback, Event, _sync_device


class EvaluationTiming(Callback):
    """Measure evaluation wall time.

    Completely data-free: it reads nothing from any event, any payload, and any
    state, only the moment each boundary happened. That is why it is a plain
    `tpen.callback.Callback` rather than a `tpen.callback.StatefulCallback`.

    Notes
    -----
    One legacy string trigger survives, deliberately. ``exception`` is a
    RUN-level event emitted by `tpen.run` when the runner raises; it has no
    typed equivalent and no owning domain, which is item ``39eacd99`` and not
    this migration. It is also the only thing that ever writes
    ``eval/perf {failed: True}``, because the evaluation domain has no
    suite-level failure moment to attach a typed event to -- a failed suite is a
    status field, and minting an event to carry it is what ADR-E007 forbids.
    Dropping the trigger would therefore silently delete a published metric
    series (ADR-E006). Ugly, and correct.

    Parameters
    ----------
    cuda_synchronize : bool, optional
        Synchronize the accelerator at both boundaries for device timing.
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
        # Importing ``tpen.callback.timing`` must stay torch-free, and importing
        # anything from `tpen.evaluation` runs that package's ``__init__``, which
        # pulls in torch. Resolve the evaluation-owned event types only when this
        # callback is constructed -- the same reason `TrainPhaseTiming` defers
        # its training imports.
        from tpen.evaluation.events import EvaluationCompleted, EvaluationStarted

        super().__init__(
            triggers=("exception",),
            typed_groups=(
                SubscriptionGroup(
                    selectors=(
                        Subscription.of(EvaluationStarted),
                        Subscription.of(EvaluationCompleted),
                    )
                ),
            ),
            **kwargs,
        )
        self.cuda_synchronize = bool(cuda_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._started_type = EvaluationStarted
        self._completed_type = EvaluationCompleted
        self._start: float | None = None

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Start the clock at the suite's start and report at its completion."""

        event = occurrence.event
        if isinstance(event, self._started_type):
            self._start_timing()
            return
        if isinstance(event, self._completed_type):
            self._log_end(context, failed=False)

    def on_exception(self, event: Event) -> None:
        """Log elapsed evaluation time on failure when evaluation had started."""

        self._log_end(event.context, failed=True)

    def _start_timing(self) -> None:
        _sync_device(self.cuda_synchronize)
        self._start = self.clock()

    def _log_end(self, context: RunContext, *, failed: bool) -> None:
        if self._start is None:
            return
        _sync_device(self.cuda_synchronize)
        metrics: dict[str, float | bool] = {"wall_time_sec": self.clock() - self._start}
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
