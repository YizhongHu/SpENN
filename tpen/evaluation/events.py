"""Typed events and operations owned by the evaluation domain.

This is the evaluation domain's counterpart to `tpen.training.events`. It
replaced a set of payload builders that flattened already-typed frozen objects
into mappings, which four callbacks then re-parsed with seventeen ``.get()``
probes and eight ``isinstance(dict)`` guards -- the magic payload dict ADR-E007
rejects, in production on the legacy side. Those builders, the fourteen string
emit sites that called them, and every probe are gone; the typed vocabulary
below is now the evaluation domain's only reporting channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from tpen.evaluation.results import EvaluationFailure
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

    Emitted only on the success path. There is deliberately no suite-level
    failure event: a failed suite is a *status field* on
    `tpen.evaluation.results.EvaluationResult`, and no distinct moment exists to
    attach one to. Minting an event so a subscriber could read that status is
    precisely the manufacturing ADR-E007 forbids.

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


__all__ = [
    "CalculatorRun",
    "ComponentFailed",
    "ComponentRun",
    "EvaluationCompleted",
    "EvaluationStarted",
    "EvaluationTaskRun",
    "GeneratorRun",
    "SummaryRun",
]
