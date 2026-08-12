"""Whole-run timing callback."""

from __future__ import annotations

import time
from typing import Any, Callable

from tpen.artifacts import RunContext
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.run_events import RunCompleted, RunFailed, RunStarted

from ..cadence import SubscriptionGroup
from .base import Callback, _sync_device


class RunTiming(Callback):
    """Measure whole-run timestamps and wall-clock duration.

    Data-free, so a plain `tpen.callback.Callback`: it reads only the moment each
    boundary happened.

    Notes
    -----
    Like `tpen.callback.ResourceUsage`, this used to log ``runtime`` TWICE on a
    failed run whenever its shipped default ``triggers`` were left in place:
    ``run_failed`` and ``exception`` are one moment under two names, and
    ``on_run_failed`` was an alias for ``on_exception``. The second record
    carried a slightly later ``end_time_unix`` and a slightly longer
    ``wall_time_sec``, so the duplicate was not even byte-identical. One typed
    `tpen.run_events.RunFailed` collapses it. No metric name changes.

    Runtime timing is logger-owned and is not an inter-callback contract.

    Parameters
    ----------
    log_start_end_timestamps : bool, optional
        Whether to log ``start_time_unix`` and ``end_time_unix``.
    log_wall_time : bool, optional
        Whether to log ``wall_time_sec``.
    accelerator_synchronize : bool, optional
        Synchronize the accelerator at each boundary for device timing.
    clock : callable, optional
        Monotonic clock override for deterministic tests.
    wall_clock : callable, optional
        Wall-clock override for deterministic tests.
    **kwargs
        Forwarded to `tpen.callback.Callback`.
    """

    def __init__(
        self,
        *,
        log_start_end_timestamps: bool = True,
        log_wall_time: bool = True,
        accelerator_synchronize: bool = False,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            typed_groups=(
                SubscriptionGroup(
                    selectors=(
                        Subscription.of(RunStarted),
                        Subscription.of(RunCompleted),
                        Subscription.of(RunFailed),
                    )
                ),
            ),
            **kwargs,
        )
        self.log_start_end_timestamps = bool(log_start_end_timestamps)
        self.log_wall_time = bool(log_wall_time)
        self.accelerator_synchronize = bool(accelerator_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self.wall_clock = time.time if wall_clock is None else wall_clock
        self._start_perf: float | None = None

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Start the run clock, and report elapsed time at either end."""

        event = occurrence.event
        if isinstance(event, RunStarted):
            self._start(context)
            return
        if isinstance(event, RunCompleted):
            self._log_end(context, failed=event.status == "failed")
            return
        if isinstance(event, RunFailed):
            self._log_end(context, failed=True)

    def _start(self, context: RunContext) -> None:
        _sync_device(self.accelerator_synchronize)
        self._start_perf = self.clock()
        if self.log_start_end_timestamps:
            context.log({"start_time_unix": self.wall_clock()}, step=0, namespace="runtime")

    def _log_end(self, context: RunContext, *, failed: bool) -> None:
        _sync_device(self.accelerator_synchronize)
        now = self.clock()
        metrics: dict[str, float | bool] = {}
        if self.log_start_end_timestamps:
            metrics["end_time_unix"] = self.wall_clock()
        if self.log_wall_time and self._start_perf is not None:
            metrics["wall_time_sec"] = now - self._start_perf
        if failed:
            metrics["failed"] = True
        if metrics:
            context.log(metrics, step=0, namespace="runtime")

    def _reset_typed_state(self) -> None:
        """Drop a half-open measurement when the owning RunContext changes."""

        self._start_perf = None


__all__ = ["RunTiming"]
