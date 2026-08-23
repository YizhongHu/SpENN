"""Evaluation component timing callback."""

from __future__ import annotations

import time
from typing import Any, Callable

from tpen.artifacts import RunContext
from tpen.events import Ended
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Operation, Started, ended, started

from ..cadence import SubscriptionGroup
from .base import Callback, TimingSource, _occurrence_time, _sync_device


class EvaluationComponentTiming(Callback):
    """Measure per-component durations within each evaluation task.

    Every component the evaluator runs is a typed
    `tpen.evaluation.events.ComponentRun` scope nested inside an
    `tpen.evaluation.events.EvaluationTaskRun` scope, so this callback is
    trigger-free: it observes ``Started``/``Ended`` boundaries instead of named
    legacy events. Durations accumulate per task and are logged as a single
    ``eval/perf/<task>`` record when the task scope ends, one key per component
    observed in that task: ``generator_time_sec``,
    ``calculator/<name>_time_sec``, and ``summary/<name>_time_sec``.

    Each component kind owns its durable metric fragment as a
    ``component_kind`` ClassVar (ADR-E006), so no metric-name literal survives
    here and renaming an operation class cannot silently rename a published
    metric. The generator's key is deliberately FLAT rather than
    ``generator/<name>_time_sec``: a task has exactly one generator, that shape
    is what is already published, and the special case is selected by TYPE, not
    by comparing the fragment to a string.

    This callback needs the task identity that `ComponentRun` deliberately does
    not carry, and reads it from the enclosing task scope. It takes it from the
    typed operation -- never from a state object, whose fields are stale at any
    boundary above their assignment -- which is why it needs no domain state at
    all and stays a plain `tpen.callback.Callback`.

    Parameters
    ----------
    accelerator_synchronize : bool, optional
        Synchronize the accelerator at component boundaries for device timing.
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
        # Deferred so that importing ``tpen.callback.timing`` stays torch-free;
        # see `EvaluationTiming` for the full reason.
        from tpen.evaluation.events import ComponentRun, EvaluationTaskRun, GeneratorRun

        super().__init__(
            typed_groups=(
                # ONE group for all four selectors. `EvaluationTaskRun` is not a
                # `ComponentRun`, so these do not overlap -- but splitting the
                # `ComponentRun` base from one of its subclasses across two
                # groups WOULD, and `validate_subscription_groups` rejects that
                # at construction (ADR-E002), which is run-killing. This
                # callback needs only the base class, so one group covers it.
                SubscriptionGroup(
                    selectors=(
                        started(EvaluationTaskRun),
                        ended(EvaluationTaskRun),
                        started(ComponentRun),
                        ended(ComponentRun),
                    )
                ),
            ),
            **kwargs,
        )
        self.accelerator_synchronize = bool(accelerator_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._timing = TimingSource(clock=self.clock, backend=timing_backend, device_backend=device_backend)
        self._task_run_type = EvaluationTaskRun
        self._component_run_type = ComponentRun
        self._generator_run_type = GeneratorRun
        # Name of the task whose scope is currently open. Task scopes never
        # nest, so one slot is enough.
        self._task: str | None = None
        # Keyed by the paired scope coordinate ``(concrete type, count)`` so
        # Started and Ended always match; the value carries the owning task.
        self._starts: dict[tuple[type[object], int], tuple[str, str, tuple[Any, Any | None]]] = {}
        self._durations: dict[str, dict[str, float]] = {}

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Open, close, or flush one admitted evaluation scope boundary."""

        event = occurrence.event
        if not isinstance(event, (Started, Ended)):
            return
        operation = event.operation
        if isinstance(operation, self._task_run_type):
            if isinstance(event, Started):
                self._task = operation.name
            else:
                self._flush(operation.name, context)
                self._task = None
            return
        if not isinstance(operation, self._component_run_type):
            return
        key = (type(operation), occurrence.count)
        if isinstance(event, Started):
            self._record_start(key, operation, _occurrence_time(occurrence, self.clock))
        else:
            self._record_end(key, _occurrence_time(occurrence, self.clock))

    def _record_start(
        self, key: tuple[type[object], int], operation: Operation, timestamp: float
    ) -> None:
        """Record one component start time under its scope coordinate."""

        task = self._task
        if task is None:
            raise ValueError(
                "evaluation component timing requires an enclosing EvaluationTaskRun scope"
            )
        metric_key = self._metric_key(operation)
        _sync_device(self.accelerator_synchronize)
        self._starts[key] = (task, metric_key, self._timing.start(timestamp))

    def _record_end(self, key: tuple[type[object], int], timestamp: float) -> None:
        """Accumulate one component duration under its task."""

        start_record = self._starts.pop(key, None)
        if start_record is None:
            return
        _sync_device(self.accelerator_synchronize)
        task, metric_key, start = start_record
        elapsed = self._timing.elapsed(start, timestamp)
        durations = self._durations.setdefault(task, {})
        durations[metric_key] = durations.get(metric_key, 0.0) + elapsed.host
        if elapsed.device is not None:
            device_key = metric_key.replace("_time_sec", "_device_time_sec")
            durations[device_key] = durations.get(device_key, 0.0) + elapsed.device

    def _metric_key(self, operation: Operation) -> str:
        """Return the durable metric key for one component operation."""

        kind = type(operation).component_kind
        if isinstance(operation, self._generator_run_type):
            return f"{kind}_time_sec"
        name = operation.name
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{kind} timing requires a non-empty component name")
        return f"{kind}/{name}_time_sec"

    def _flush(self, task: str, context: RunContext) -> None:
        """Log the finished task's component durations as one record."""

        # A component whose Ended boundary never arrived must not leak its start
        # into the next run of the same task.
        self._starts = {
            key: value for key, value in self._starts.items() if value[0] != task
        }
        metrics = self._durations.pop(task, None)
        if not metrics:
            return
        # Evaluation's coordinate is the task namespace, a string, already
        # spelled in the metric namespace. Every evaluation record is logged at
        # step 0, written here rather than read from a state cursor.
        context.log(metrics, step=0, namespace=f"eval/perf/{task}")

    def _reset_typed_state(self) -> None:
        """Clear timing caches when the owning RunContext identity changes."""

        self._task = None
        self._starts.clear()
        self._durations.clear()


__all__ = ["EvaluationComponentTiming"]
