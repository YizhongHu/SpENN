"""Mutable evaluation-run state shared with typed callbacks."""

from __future__ import annotations

from dataclasses import dataclass

from tpen.evaluation.results import TaskResult
from tpen.events import DomainState


@dataclass
class EvaluationRunState(DomainState):
    """The evaluation domain's `tpen.events.DomainState` for one evaluator run.

    One instance lives for the whole suite and is updated in place as tasks
    complete, so its identity is stable for the lifetime of every scope that
    carries it. That stability is the delivery mechanism's single requirement
    (ADR-E008): `tpen.artifacts.RunContext.scope` captures this reference at
    entry and hands the SAME object to both boundaries, so a handler at
    ``Ended`` observes whatever the scope body wrote. Rebinding the name with
    ``dataclasses.replace`` inside a scope would leave that scope holding a
    stale object and break delivery *silently*, which is why this is a mutable
    container rather than a frozen value. `tpen.training.state.TrainerState`
    works the same way -- it is a mutable container whose fields are rebound to
    frozen replacement values -- so the two domains do not differ here.

    The two domains do differ in vocabulary: this state shares no field with
    `TrainerState`, and evaluation is coordinated by a task namespace *string*
    rather than by an integer step. That is exactly why `DomainState` is empty.

    Only one field exists, and deliberately so (ADR-E003). Every other
    candidate a suite could expose -- the task spec, the `EvaluationContext`,
    the running metric/failure/artifact lists -- has no present consumer, and
    the running *component* has no type at all: a
    `tpen.evaluation.task.EvaluationTask` holds its generator, calculators, and
    summaries as bare ``object``, so exposing one here would either reintroduce
    a ``getattr`` probe or mint a protocol nobody asked for.

    Parameters
    ----------
    task_result : TaskResult or None, optional
        Result of the task whose scope is currently open, or ``None`` before
        that task has finished. `tpen.evaluation.evaluator.Evaluator` resets it
        to ``None`` before each task's scope opens, so a handler at
        ``Started[EvaluationTaskRun]`` cannot read the previous task's result,
        and writes it inside the scope body, so a handler at
        ``Ended[EvaluationTaskRun]`` reads the current one. It is therefore also
        ``None`` at a component boundary, which fires before the task result
        exists.
    """

    task_result: TaskResult | None = None


__all__ = ["EvaluationRunState"]
