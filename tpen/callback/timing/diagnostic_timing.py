"""Diagnostic timing callback.

Importing this module resolves `tpen.evaluation`, and therefore torch, because
`DiagnosticTiming` declares its ``state_type`` as a class fact.
`tpen.callback.timing` loads it lazily so that importing the timing package
stays torch-free; the other evaluation timing callbacks resolve their event
types inside ``__init__`` instead, which a ClassVar cannot do.
"""

from __future__ import annotations

import time
from typing import Any, Callable, ClassVar

from tpen.artifacts import RunContext
from tpen.evaluation.state import EvaluationRunState
from tpen.events import DomainState, Ended
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Started, ended, started

from ..base import StatefulCallback
from ..cadence import SubscriptionGroup
from .base import _sync_device


class DiagnosticTiming(StatefulCallback[EvaluationRunState]):
    """Measure per-evaluation-task durations and report whether each failed.

    Times every `tpen.evaluation.events.EvaluationTaskRun` scope and logs one
    ``diagnostics/<task> {time_sec, failed}`` record when it ends.

    The ``failed`` flag is what makes this callback the evidence ADR-E008 asked
    D1 for: the evaluator writes the finished `tpen.evaluation.results.TaskResult`
    into the domain state as the last statement of the task scope's body, so the
    ``Ended`` boundary observes it, on the success and the failure path alike --
    an evaluation task failure is a VALUE, not a raised exception. A published
    metric key therefore depends on typed state delivery actually working. The
    legacy path instead discriminated the same moment by comparing a string,
    picking ``task_end`` or ``task_failed`` from the status.

    Notes
    -----
    The class name predates the metric it writes. The ``diagnostics/`` namespace
    is durable (ADR-E006) and the class is named in shipped configs, so neither
    is renamed here; the three ``diagnostic_*`` string handlers this class also
    carried are gone, because `tpen.diagnostics.evaluate_diagnostics` has no
    callers anywhere in the repository and those events were never emitted by
    any run.

    Parameters
    ----------
    cuda_synchronize : bool, optional
        Synchronize the accelerator at task boundaries for device timing.
    clock : callable, optional
        Monotonic clock override for deterministic tests.
    """

    state_type: ClassVar[type[DomainState]] = EvaluationRunState

    def __init__(
        self,
        *,
        cuda_synchronize: bool = False,
        clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        from tpen.evaluation.events import EvaluationTaskRun

        super().__init__(
            typed_groups=(
                SubscriptionGroup(
                    selectors=(started(EvaluationTaskRun), ended(EvaluationTaskRun))
                ),
            ),
            **kwargs,
        )
        self.cuda_synchronize = bool(cuda_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._task_run_type = EvaluationTaskRun
        # Keyed by the paired scope coordinate so Started and Ended always match.
        self._starts: dict[tuple[type[object], int], float] = {}

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: EvaluationRunState,
    ) -> None:
        """Time one evaluation task scope and report its outcome."""

        event = occurrence.event
        if not isinstance(event, (Started, Ended)):
            return
        operation = event.operation
        if not isinstance(operation, self._task_run_type):
            return
        key = (type(operation), occurrence.count)
        if isinstance(event, Started):
            _sync_device(self.cuda_synchronize)
            self._starts[key] = self.clock()
            return
        if key not in self._starts:
            return
        result = state.task_result
        if result is None:
            # `scope` fires ``Ended`` from a ``finally``, so this boundary is
            # also reached when the evaluator raises out of the task body, and
            # then no result was ever written. The legacy path emitted neither
            # ``task_end`` nor ``task_failed`` there, so no ``diagnostics/``
            # record was written either. Preserve the silence rather than
            # publishing a duration with a guessed status.
            self._starts.pop(key)
            return
        _sync_device(self.cuda_synchronize)
        metrics: dict[str, float | bool] = {"time_sec": self.clock() - self._starts.pop(key)}
        if result.failed:
            metrics["failed"] = True
        # The task identity rides the typed operation, and evaluation's step
        # coordinate is 0 -- neither is read from the state cursor.
        context.log(metrics, step=0, namespace=f"diagnostics/{operation.name}")

    def _reset_typed_state(self) -> None:
        """Clear timing caches when the owning RunContext identity changes."""

        self._starts.clear()


__all__ = ["DiagnosticTiming"]
