"""Tests for evaluator component lifecycle occurrences.

These used to assert the evaluator's legacy STRING sequence. Slice 2 of D1
deleted that path -- the four payload builders and all fourteen emit sites --
so the same control flow is asserted here through the typed vocabulary, which
is now the evaluation domain's only reporting channel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from tpen.artifacts import RunContext
from tpen.callback import Callback, SubscriptionGroup
from tpen.evaluation.events import (
    CalculatorRun,
    ComponentFailed,
    ComponentRun,
    EvaluationTaskRun,
)
from tpen.events import Ended, Occurrence, Started, Subscription, ended, started
from tests.helpers.evaluation_components import (
    FailingCalculator,
    FailingGenerator,
    single_task_evaluator,
)
from tests.helpers.run_context import make_run_context


class _OccurrenceRecorder(Callback):
    """Capture every typed evaluation occurrence, in delivery order."""

    def __init__(self) -> None:
        super().__init__(
            typed_groups=(
                # One group: `ComponentRun` and any subclass selector in separate
                # groups would be rejected as overlapping (ADR-E002).
                SubscriptionGroup(
                    selectors=(
                        started(EvaluationTaskRun),
                        ended(EvaluationTaskRun),
                        started(ComponentRun),
                        ended(ComponentRun),
                        Subscription.of(ComponentFailed),
                    ),
                ),
            )
        )
        self.seen: list[Any] = []

    def handle_occurrence_impl(self, occurrence: Occurrence[Any], context: Any) -> None:
        self.seen.append(occurrence.event)

    def labels(self) -> list[str]:
        return [
            f"{type(event).__name__}[{type(event.operation).__name__}]"
            if isinstance(event, (Started, Ended))
            else type(event).__name__
            for event in self.seen
        ]


def _run_context(run_dir: Path, recorder: _OccurrenceRecorder) -> RunContext:
    return make_run_context(run_dir, callbacks=[recorder])


def test_component_scopes_bracket_each_component_on_success(tmp_path: Path) -> None:
    recorder = _OccurrenceRecorder()
    evaluator = single_task_evaluator(tmp_path)

    result = evaluator.evaluate(model=nn.Linear(1, 1), context=_run_context(tmp_path, recorder))

    assert result.status == "success"
    assert recorder.labels() == [
        "Started[EvaluationTaskRun]",
        "Started[GeneratorRun]",
        "Ended[GeneratorRun]",
        "Started[CalculatorRun]",
        "Ended[CalculatorRun]",
        "Started[SummaryRun]",
        "Ended[SummaryRun]",
        "Ended[EvaluationTaskRun]",
    ]
    # Each component operation carries its own instance name; the owning task is
    # read from the enclosing task scope rather than repeated on every event.
    assert [
        event.operation.name
        for event in recorder.seen
        if isinstance(event, Started) and isinstance(event.operation, ComponentRun)
    ] == ["null", "identity", "metric"]


def test_calculator_scope_closes_before_the_failure_is_reported(tmp_path: Path) -> None:
    recorder = _OccurrenceRecorder()
    evaluator = single_task_evaluator(tmp_path, calculators=[FailingCalculator()])

    result = evaluator.evaluate(model=nn.Linear(1, 1), context=_run_context(tmp_path, recorder))

    assert result.status == "failed"
    # `scope` emits ``Ended`` from a ``finally``, so the component scope closes
    # even though the calculator raised. The default "continue" policy still
    # runs summaries afterwards.
    assert recorder.labels() == [
        "Started[EvaluationTaskRun]",
        "Started[GeneratorRun]",
        "Ended[GeneratorRun]",
        "Started[CalculatorRun]",
        "Ended[CalculatorRun]",
        "ComponentFailed",
        "Started[SummaryRun]",
        "Ended[SummaryRun]",
        "Ended[EvaluationTaskRun]",
    ]
    failed = [event for event in recorder.seen if isinstance(event, ComponentFailed)]
    assert failed[0].failure.component == "broken"
    assert failed[0].failure.error_type == "RuntimeError"


def test_generator_failure_reports_the_component_before_the_task_boundary(tmp_path: Path) -> None:
    """The legacy path's inverted generator order does not survive typing.

    On the string path this case emitted ``task_failed`` BEFORE
    ``generator_failed`` -- the reverse of the calculator and summary paths --
    because the generator branch had to build the task result in order to return
    it. `test_generator_failure_emits_task_failed_first` pinned that asymmetry so
    a rewrite could not flatten it by accident.

    Deleting the string path does not flatten the asymmetry; it DISSOLVES it.
    There is no typed task-failure event to order: the task outcome is delivered
    at ``Ended[EvaluationTaskRun]``, which `scope` fires from a ``finally`` and
    which is therefore structurally last on every path. So all three component
    kinds now report the failure first, uniformly. This test pins the new
    uniform order, and the state assertions below pin why it is uniform.
    """

    recorder = _OccurrenceRecorder()
    evaluator = single_task_evaluator(tmp_path, generator=FailingGenerator())

    result = evaluator.evaluate(model=nn.Linear(1, 1), context=_run_context(tmp_path, recorder))

    assert result.status == "failed"
    assert recorder.labels() == [
        "Started[EvaluationTaskRun]",
        "Started[GeneratorRun]",
        "Ended[GeneratorRun]",
        "ComponentFailed",
        "Ended[EvaluationTaskRun]",
    ]
    failed = [event for event in recorder.seen if isinstance(event, ComponentFailed)]
    assert failed[0].failure.component == "broken-generator"
    assert failed[0].failure.component_type == "generator"
    # No calculator or summary runs once the generator has failed.
    assert not any(
        isinstance(event, (Started, Ended)) and isinstance(event.operation, CalculatorRun)
        for event in recorder.seen
    )
