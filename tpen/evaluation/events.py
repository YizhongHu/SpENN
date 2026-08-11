"""Typed events and operations owned by the evaluation domain.

The typed vocabulary below is the evaluation domain's counterpart to
`tpen.training.events`. The legacy string-payload builders at the bottom of
this module are what it replaces: each one flattens an already-typed frozen
object into a mapping that four callbacks then re-parse with ``.get()`` probes
and ``isinstance(dict)`` guards. That is the magic payload dict ADR-E007
rejects, so the builders are scheduled for deletion once the callbacks move
onto the typed events. Until then both are emitted, side by side, and no
callback has been migrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from tpen.evaluation.results import EvaluationFailure, TaskResult
from tpen.evaluation.task import EvaluationTask
from tpen.events import Event, Operation


@dataclass(frozen=True)
class EvaluationTaskRun(Operation):
    """One evaluation task's generator/calculator/summary pipeline.

    Parameters
    ----------
    name : str
        Configured task name.
    namespace : str
        Metric namespace the task's metrics are logged under. This is the
        evaluation domain's coordinate, in place of the training domain's
        integer step.
    output_dir : pathlib.Path
        Resolved task-local artifact directory.
    """

    name: str
    namespace: str
    output_dir: Path


@dataclass(frozen=True)
class ComponentRun(Operation):
    """One component invocation inside an evaluation task.

    Concrete subclasses declare ``component_kind``, the durable metric fragment
    that `tpen.callback.timing.EvaluationComponentTiming` turns into keys such
    as ``generator_time_sec``. Per ADR-E006 that fragment rides the type, so a
    class rename cannot silently rename a published metric and no separate name
    table can drift out of sync -- the same idiom as
    `tpen.training.events.TrainingPhase.phase_name`. The component *instance's*
    name is data rather than a class fact, so it is a field.

    This base leaves ``component_kind`` unset and is therefore abstract:
    constructing it, or a subclass that forgot to declare one, fails loudly
    instead of silently producing an unnamed timing metric.

    A `ComponentRun` scope is always nested inside an `EvaluationTaskRun` scope,
    which is where a subscriber reads the owning task from. The task identity is
    deliberately not repeated here.

    Notes
    -----
    A single callback must not select `ComponentRun` in one subscription group
    and one of its subclasses in another:
    `tpen.callback.cadence.validate_subscription_groups` rejects overlapping
    selectors by ``issubclass``, and that rejection is run-killing.

    Parameters
    ----------
    name : str or None
        Name of the component instance, or ``None`` when it has none.

    Raises
    ------
    TypeError
        If the concrete type declares no ``component_kind``.
    """

    # ClassVar: neither a dataclass field nor a serialized occurrence field.
    component_kind: ClassVar[str]

    name: str | None

    def __post_init__(self) -> None:
        # A component with no durable metric fragment cannot be timed, so reject
        # it at construction rather than at metric-key lookup.
        if not hasattr(type(self), "component_kind"):
            raise TypeError(
                f"{type(self).__name__} must declare a component_kind ClassVar; "
                "ComponentRun itself is abstract"
            )


@dataclass(frozen=True)
class GeneratorRun(ComponentRun):
    """Produce the configurations one evaluation task consumes."""

    component_kind: ClassVar[str] = "generator"


@dataclass(frozen=True)
class CalculatorRun(ComponentRun):
    """Derive one evaluation task's intermediate quantities."""

    component_kind: ClassVar[str] = "calculator"


@dataclass(frozen=True)
class SummaryRun(ComponentRun):
    """Reduce one evaluation task's bundle to metrics and artifacts."""

    component_kind: ClassVar[str] = "summary"


@dataclass(frozen=True)
class EvaluationStarted(Event):
    """The evaluation suite is about to run its configured tasks.

    Paired in meaning with `EvaluationCompleted`, but deliberately NOT modelled
    as an `Operation` scope. `tpen.artifacts.RunContext.scope` emits its
    ``Ended`` record from a ``finally``, so a scope would fire ``Ended`` when
    the evaluator raises. Today a raising evaluator instead reaches the
    run-level ``exception`` event, which is what makes
    `tpen.callback.timing.EvaluationTiming` write ``eval/perf`` with
    ``failed: True``; an ``Ended`` arriving first would consume that callback's
    start timestamp and the later ``exception`` would return early, deleting the
    ``eval/perf/failed`` series. A durable metric may not be dropped by a
    refactor (ADR-E006), so these stay point events.

    The suite carries no state: `EvaluationRunState` holds a per-task result,
    which has no meaning at a suite boundary.
    """


@dataclass(frozen=True)
class EvaluationCompleted(Event):
    """The evaluation suite ran every task it was going to run and returned.

    Emitted only on the success path, mirroring the legacy ``evaluate_end``
    string. There is deliberately no suite-level failure event: a failed suite
    is a *status field* on `tpen.evaluation.results.EvaluationResult`, and no
    distinct moment exists to attach one to. Minting an event so a subscriber
    could read that status is precisely the manufacturing ADR-E007 forbids.

    See `EvaluationStarted` for why this is a point event and not a scope.
    """


@dataclass(frozen=True)
class ComponentFailed(Event):
    """One evaluation component failed, or could not be run at all.

    Covers a raising generator, calculator, or summary and a summary whose
    required bundle fields are absent. The failing component's kind is already
    on ``failure``, so no per-kind subclass exists.

    Parameters
    ----------
    failure : EvaluationFailure
        Structured failure record for the component. This is the typed object
        the legacy ``<kind>_failed`` payload flattened with ``to_dict()``.
    """

    failure: EvaluationFailure


# --------------------------------------------------------------------------
# Legacy string-event payload builders, replaced by the typed vocabulary above
# and kept only until the evaluation callbacks are migrated off the strings.
# --------------------------------------------------------------------------


def task_payload(task: EvaluationTask, *, output_dir: str | Path | None = None) -> dict[str, object]:
    """Return the standard payload for task lifecycle events."""

    payload: dict[str, object] = {
        "task_name": task.name,
        "task_namespace": task.namespace,
    }
    if output_dir is not None:
        payload["output_dir"] = str(output_dir)
    return payload


def component_payload(
    *,
    task: EvaluationTask,
    component_name: str | None,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Return the standard payload for component lifecycle events."""

    return {
        **task_payload(task, output_dir=output_dir),
        "component_name": component_name,
    }


def component_failure_payload(
    *,
    task: EvaluationTask,
    component_name: str | None,
    failure: EvaluationFailure,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Return a standard component-failure event payload."""

    return {
        **task_payload(task, output_dir=output_dir),
        "component_name": component_name,
        "failure": failure.to_dict(),
    }


def task_result_payload(task_result: TaskResult) -> dict[str, object]:
    """Return the standard payload for task completion/failure events."""

    return {"task_result": task_result.to_payload()}


__all__ = [
    "CalculatorRun",
    "ComponentFailed",
    "ComponentRun",
    "EvaluationCompleted",
    "EvaluationStarted",
    "EvaluationTaskRun",
    "GeneratorRun",
    "SummaryRun",
    "component_failure_payload",
    "component_payload",
    "task_payload",
    "task_result_payload",
]
