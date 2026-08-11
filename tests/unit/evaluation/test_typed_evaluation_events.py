"""Tests for the evaluation domain's typed event vocabulary.

Slice 1 of D1 lands the vocabulary purely additively: the typed occurrences are
emitted beside the legacy string events and no callback has been migrated, so
these tests assert what the evaluator now *emits* and what an evaluation-domain
`tpen.callback.StatefulCallback` would *receive*, not any production callback's
behaviour.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch
from torch import nn

from tpen.callback import Callback, StatefulCallback, SubscriptionGroup
from tpen.checkpoint import CheckpointRestored, RestoreReport
from tpen.data.batch import ElectronBatch
from tpen.evaluation import EvaluationTask, Evaluator
from tpen.evaluation.bundle import EvaluationBundle, GeneratedConfigurations
from tpen.evaluation.events import (
    CalculatorRun,
    ComponentFailed,
    ComponentRun,
    EvaluationCompleted,
    EvaluationStarted,
    EvaluationTaskRun,
    GeneratorRun,
    SummaryRun,
)
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.results import SummaryResult, TaskResult
from tpen.evaluation.state import EvaluationRunState
from tpen.events import DomainState, Ended, Occurrence, Started, Subscription, ended, started
from tpen.training.state import TrainerState
from tests.helpers.run_context import make_run_context


class _NullGenerator:
    name = "null"

    def generate(self, *, model: nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        batch = ElectronBatch(
            positions=torch.zeros(1, 2, 3, dtype=torch.float64),
            spins=torch.tensor([[1.0, -1.0]], dtype=torch.float64),
        )
        return GeneratedConfigurations(batch=batch, metadata={})


class _FailingGenerator:
    name = "broken-generator"

    def generate(self, *, model: nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        raise RuntimeError("generator boom")


class _IdentityCalculator:
    name = "identity"

    def calculate(self, *, model: nn.Module | None, bundle: EvaluationBundle, context: EvaluationContext) -> EvaluationBundle:
        return bundle


class _MetricSummary:
    name = "metric"
    required_fields: frozenset[str] = frozenset()

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        return SummaryResult(metrics={"value": 1.0})


class _OccurrenceRecorder(Callback):
    """Plain callback capturing every typed evaluation occurrence, in order."""

    def __init__(self) -> None:
        super().__init__(
            typed_groups=(
                SubscriptionGroup(
                    selectors=(
                        started(EvaluationTaskRun),
                        ended(EvaluationTaskRun),
                        started(ComponentRun),
                        ended(ComponentRun),
                        Subscription.of(EvaluationStarted),
                        Subscription.of(EvaluationCompleted),
                        Subscription.of(ComponentFailed),
                        Subscription.of(CheckpointRestored),
                    ),
                ),
            )
        )
        self.seen: list[Any] = []

    def handle_occurrence_impl(self, occurrence: Occurrence[Any], context: Any) -> None:
        self.seen.append(occurrence.event)

    def labels(self) -> list[str]:
        """Return one readable label per observed occurrence."""

        return [_label(event) for event in self.seen]


class _StateRecorder(StatefulCallback[EvaluationRunState]):
    """Evaluation-domain stateful callback capturing (label, task_result) pairs."""

    state_type: ClassVar[type[DomainState]] = EvaluationRunState

    def __init__(self, *groups: SubscriptionGroup) -> None:
        super().__init__(typed_groups=groups)
        self.seen: list[tuple[str, TaskResult | None]] = []
        self.states: list[EvaluationRunState] = []

    def handle_occurrence_impl(
        self, occurrence: Occurrence[Any], context: Any, state: EvaluationRunState
    ) -> None:
        self.seen.append((_label(occurrence.event), state.task_result))
        self.states.append(state)


def _label(event: Any) -> str:
    """Render one typed event as ``Boundary[Type]`` or ``Type``."""

    if isinstance(event, (Started, Ended)):
        return f"{type(event).__name__}[{type(event.operation).__name__}]"
    return type(event).__name__


def _evaluator(output_dir: Path, *, generator: object, summaries: list[object] | None = None) -> Evaluator:
    return Evaluator(
        namespace="eval",
        tasks=[
            EvaluationTask(
                name="energy",
                namespace="eval/energy",
                output_dir=output_dir,
                generator=generator,
                calculators=[_IdentityCalculator()],
                summaries=[_MetricSummary()] if summaries is None else summaries,
            )
        ],
    )


def _noop_emit(name: str, *, payload: dict[str, Any] | None = None) -> None:
    """Swallow the legacy string events these tests do not assert on."""

    del name, payload


# --------------------------------------------------------------------------
# The vocabulary itself
# --------------------------------------------------------------------------


def test_component_kind_matches_the_legacy_event_name_prefix() -> None:
    """Pin each ``component_kind`` to the string fragment it will replace.

    `Evaluator` still spells the legacy ``<kind>_start``/``<kind>_end`` prefix as
    its own literal, so slice 2 can only collapse the two spellings into one if
    they agree today. ADR-E006 puts that durable fragment on the type.
    """

    assert GeneratorRun.component_kind == "generator"
    assert CalculatorRun.component_kind == "calculator"
    assert SummaryRun.component_kind == "summary"


def test_component_run_base_is_abstract() -> None:
    """A component with no durable metric fragment cannot be timed."""

    with pytest.raises(TypeError, match="component_kind"):
        ComponentRun(name="anything")


def test_component_kinds_are_class_facts_and_the_name_is_a_field() -> None:
    """``component_kind`` must not be a dataclass field, and so not serialized."""

    assert [field.name for field in fields(GeneratorRun)] == ["name"]
    assert GeneratorRun(name="a") == GeneratorRun(name="a")
    assert GeneratorRun(name="a") != CalculatorRun(name="a")


def test_a_callback_cannot_select_both_component_run_and_a_subclass() -> None:
    """ADR-E002: overlapping selectors across groups are rejected, loudly.

    Splitting `ComponentRun` and one of its subclasses across two cadence groups
    is the mistake this hierarchy invites, and it is run-killing rather than
    silently duplicating deliveries.
    """

    with pytest.raises(ValueError, match="overlapping deliveries"):
        _StateRecorder(
            SubscriptionGroup(selectors=(started(ComponentRun),)),
            SubscriptionGroup(selectors=(started(CalculatorRun),)),
        )


# --------------------------------------------------------------------------
# The state object
# --------------------------------------------------------------------------


def test_evaluation_run_state_shares_no_field_with_the_training_state() -> None:
    """The two domains agree on nothing, which is why `DomainState` is empty."""

    evaluation_fields = {field.name for field in fields(EvaluationRunState)}
    training_fields = {field.name for field in fields(TrainerState)}
    assert evaluation_fields == {"task_result"}
    assert evaluation_fields.isdisjoint(training_fields)


def test_evaluation_run_state_is_mutable_so_scope_delivery_survives() -> None:
    """A frozen state rebound with ``replace`` would break `scope` silently."""

    state = EvaluationRunState()
    state.task_result = None  # assignable: not frozen
    assert isinstance(state, DomainState)


def test_the_task_ended_boundary_observes_the_result_written_in_the_body(tmp_path: Path) -> None:
    """Executable proof that the evaluator's state delivery actually works.

    `scope` captures the state *reference* at entry and hands the same object to
    both boundaries, so the result the evaluator writes inside the task body is
    visible at ``Ended`` -- and is still absent at ``Started``.
    """

    recorder = _StateRecorder(
        SubscriptionGroup(selectors=(started(EvaluationTaskRun), ended(EvaluationTaskRun)))
    )
    context = make_run_context(tmp_path, callbacks=[recorder])
    evaluator = _evaluator(tmp_path / "energy", generator=_NullGenerator())

    result = evaluator.evaluate(model=nn.Linear(1, 1), context=context, emit=_noop_emit)

    assert result.status == "success"
    assert [label for label, _ in recorder.seen] == [
        "Started[EvaluationTaskRun]",
        "Ended[EvaluationTaskRun]",
    ]
    assert recorder.seen[0][1] is None
    assert recorder.seen[1][1] == result.task_results[0]
    # One state object for the whole suite: identity is what `scope` relies on.
    assert recorder.states[0] is recorder.states[1]


def test_a_later_task_start_does_not_observe_the_previous_task_result(tmp_path: Path) -> None:
    """The suite-long state is cleared at each task entry, not left stale."""

    recorder = _StateRecorder(
        SubscriptionGroup(selectors=(started(EvaluationTaskRun), ended(EvaluationTaskRun)))
    )
    context = make_run_context(tmp_path, callbacks=[recorder])
    evaluator = Evaluator(
        namespace="eval",
        tasks=[
            EvaluationTask(
                name=name,
                namespace=f"eval/{name}",
                output_dir=tmp_path / name,
                generator=_NullGenerator(),
                calculators=[],
                summaries=[_MetricSummary()],
            )
            for name in ("first", "second")
        ],
    )

    result = evaluator.evaluate(model=nn.Linear(1, 1), context=context, emit=_noop_emit)

    observed = [(label, None if task is None else task.name) for label, task in recorder.seen]
    assert observed == [
        ("Started[EvaluationTaskRun]", None),
        ("Ended[EvaluationTaskRun]", "first"),
        ("Started[EvaluationTaskRun]", None),
        ("Ended[EvaluationTaskRun]", "second"),
    ]
    assert [task.name for task in result.task_results] == ["first", "second"]


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------


def test_a_successful_task_emits_nested_typed_scopes(tmp_path: Path) -> None:
    """Component scopes nest inside the task scope, in component order."""

    recorder = _OccurrenceRecorder()
    context = make_run_context(tmp_path, callbacks=[recorder])
    evaluator = _evaluator(tmp_path / "energy", generator=_NullGenerator())

    evaluator.evaluate(model=nn.Linear(1, 1), context=context, emit=_noop_emit)

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


def test_the_task_operation_carries_the_task_identity(tmp_path: Path) -> None:
    recorder = _OccurrenceRecorder()
    context = make_run_context(tmp_path, callbacks=[recorder])
    evaluator = _evaluator(tmp_path / "energy", generator=_NullGenerator())

    evaluator.evaluate(model=nn.Linear(1, 1), context=context, emit=_noop_emit)

    task_run = recorder.seen[0].operation
    assert task_run == EvaluationTaskRun(
        name="energy", namespace="eval/energy", output_dir=tmp_path / "energy"
    )
    # The component operation carries only its own name; the owning task is read
    # from the enclosing task scope.
    assert recorder.seen[1].operation == GeneratorRun(name="null")


def test_a_failing_component_emits_the_typed_failure_object(tmp_path: Path) -> None:
    """`ComponentFailed` carries `EvaluationFailure` itself, not a mapping."""

    recorder = _OccurrenceRecorder()
    context = make_run_context(tmp_path, callbacks=[recorder])
    evaluator = _evaluator(tmp_path / "energy", generator=_FailingGenerator())

    result = evaluator.evaluate(model=nn.Linear(1, 1), context=context, emit=_noop_emit)

    assert result.status == "failed"
    failed = [event for event in recorder.seen if isinstance(event, ComponentFailed)]
    assert len(failed) == 1
    assert failed[0].failure == result.task_results[0].failures[0]
    assert failed[0].failure.component_type == "generator"
    assert failed[0].failure.error_type == "RuntimeError"
    # The generator scope still closes before the task scope, even on failure.
    assert recorder.labels() == [
        "Started[EvaluationTaskRun]",
        "Started[GeneratorRun]",
        "Ended[GeneratorRun]",
        "ComponentFailed",
        "Ended[EvaluationTaskRun]",
    ]


def test_checkpoint_restored_reaches_the_durable_record_field_wise(tmp_path: Path) -> None:
    """The report's fields must survive serialization, not collapse to a marker.

    `CheckpointRestored` has no callback subscriber; the durable record IS its
    consumer, so ``completed_updates`` reaching ``occurrences.jsonl`` is the
    whole reason the event exists.
    """

    context = make_run_context(tmp_path)

    context.emit(
        CheckpointRestored(
            report=RestoreReport(
                mode="model_only",
                checkpoint_dir="ckpt/step_000010",
                schema_version=2,
                next_iteration=11,
                completed_updates=10,
                loaded_model=True,
            )
        )
    )

    records = [
        json.loads(line)
        for line in context.path("occurrences.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["event"].endswith("CheckpointRestored")
    assert records[0]["fields"]["report"]["completed_updates"] == 10
    assert records[0]["fields"]["report"]["next_iteration"] == 11
    assert records[0]["fields"]["report"]["mode"] == "model_only"
