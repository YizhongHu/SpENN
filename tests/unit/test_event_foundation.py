"""Focused tests for the typed-event foundation and sampling pilot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from typeguard import suppress_type_checks

from tpen.artifacts import (
    ArtifactManager,
    RunClock,
    RunContext,
    RunMetadata,
)
from tpen.checkpoint.restore import RestoreReport
from tpen.callback import TrainPhaseTiming
from tpen.events import Ended, Event, Occurrence, Operation, Started, Subscription
from tpen.events import ended, started
from tpen.training.events import (
    Backward,
    CollectSamples,
    TrainingIteration,
    TrainingIterationCompleted,
    TrainingPhase,
    UpdateCompleted,
    UpdateSkipped,
)


@dataclass(frozen=True)
class _Pulse(Event):
    label: str


@dataclass(frozen=True)
class _OtherPulse(Event):
    label: str


@dataclass(frozen=True)
class _Work(Operation):
    label: str


class _MarkerEvent(Event):
    pass


class _MarkerOperation(Operation):
    pass


class _StatefulEvent(Event):
    def __init__(self) -> None:
        self.value = "state"


class _StatefulOperation(Operation):
    def __init__(self) -> None:
        self.value = "state"


class _SlottedStatefulEvent(Event):
    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = "state"


@dataclass(frozen=True)
class _PrivatePulse(Event):
    label: str
    _private: str


@dataclass(frozen=True)
class _Detail:
    """A plain domain dataclass: neither an ``Event`` nor an ``Operation``."""

    code: int
    note: str
    _hidden: str = "hidden"


@dataclass(frozen=True)
class _Wrapper:
    """A plain dataclass whose own field is another plain dataclass."""

    detail: _Detail
    label: str


@dataclass(frozen=True)
class _ReportPulse(Event):
    """Stands in for D1's ``CheckpointRestored(report: RestoreReport)``."""

    report: RestoreReport


@dataclass(frozen=True)
class _WrappedPulse(Event):
    wrapper: _Wrapper


@dataclass(frozen=True)
class _ContainerPulse(Event):
    details: tuple[_Detail, ...]
    by_name: dict[str, _Detail]


@dataclass(frozen=True)
class _TensorPulse(Event):
    tensor: torch.Tensor


@dataclass
class _Cyclic:
    """Mutable so a field can be rebound to point back at its own parent."""

    child: object = None


@dataclass(frozen=True)
class _CyclicPulse(Event):
    node: _Cyclic


@dataclass(frozen=True)
class _SiblingPulse(Event):
    """One dataclass instance bound to two sibling fields of the same parent."""

    left: _Detail
    right: _Detail


@dataclass(frozen=True)
class _DiamondPulse(Event):
    """One dataclass instance reachable at two different depths."""

    wrapper: _Wrapper
    detail: _Detail


@dataclass(frozen=True)
class _Link:
    """One node of a deep *acyclic* chain of distinct dataclass instances."""

    depth: int
    child: object = None


@dataclass(frozen=True)
class _ChainPulse(Event):
    link: _Link


class _Recorder:
    def __init__(self, name: str, order: list[tuple[str, Occurrence[Any]]]) -> None:
        self.name = name
        self.order = order
        self.occurrences: list[Occurrence[Any]] = []
    def handle_occurrence(self, occurrence: Occurrence[Any], context: RunContext) -> None:
        del context
        self.occurrences.append(occurrence)
        self.order.append((self.name, occurrence))


class _RaisesOnStarted:
    def __init__(self) -> None:
        self.occurrences: list[Occurrence[Any]] = []

    def handle_occurrence(self, occurrence: Occurrence[Any], context: RunContext) -> None:
        del context
        self.occurrences.append(occurrence)
        if isinstance(occurrence.event, Started):
            raise RuntimeError("started dispatch failed")


class _Logger:
    def __init__(self) -> None:
        self.records: list[Any] = []

    def log(self, record: Any) -> None:
        self.records.append(record)


class _RaisingLogger:
    def log(self, record: Any) -> None:
        del record
        raise RuntimeError("timing report failed")


class _Clock:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        return self.values.pop(0)


def test_subscription_factories_match_typed_subjects_by_isinstance() -> None:
    class _BaseWork(Operation):
        pass

    class _ChildWork(_BaseWork):
        pass

    operation = _ChildWork()

    assert Subscription.of(Event).matches(_Pulse("plain"))
    assert not Subscription.of(Event).matches(Started(operation))
    assert Subscription.started(_BaseWork).matches(Started(operation))
    assert Subscription.ended(_BaseWork).matches(Ended(operation))
    assert started(_BaseWork) == Subscription.started(_BaseWork)
    assert ended(_BaseWork) == Subscription.ended(_BaseWork)
    assert not started(_BaseWork).matches(Ended(operation))


def test_subscription_validation_rejects_invalid_runtime_shapes() -> None:
    with suppress_type_checks():
        with pytest.raises(TypeError, match="subject"):
            Subscription.of("event")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Started"):
            Subscription.of(Started)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="lifecycle=None"):
            Subscription(subject=_Pulse, lifecycle=Started)
        with pytest.raises(ValueError, match="require bare"):
            Subscription(subject=_Work)
        with pytest.raises(ValueError, match="require bare"):
            Subscription(subject=_Work, lifecycle=_Pulse)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="require bare"):
            Subscription(subject=_Work, lifecycle=Started[_Work])  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="subject"):
            Subscription.of(Started[_Work])  # type: ignore[arg-type]


def test_emit_counts_by_concrete_type_and_dispatches_in_callback_order(tmp_path: Path) -> None:
    order: list[tuple[str, Occurrence[Any]]] = []
    first = _Recorder("first", order)
    second = _Recorder("second", order)
    context = _context(tmp_path, callbacks=[first, second])

    pulse_one = context.emit(_Pulse("one"))
    other = context.emit(_OtherPulse("other"))
    pulse_two = context.emit(_Pulse("two"))
    fresh_context = _context(tmp_path / "fresh-run")
    fresh_pulse = fresh_context.emit(_Pulse("fresh"))

    assert (pulse_one.count, other.count, pulse_two.count) == (1, 1, 2)
    assert fresh_pulse.count == 1
    assert order == [
        ("first", pulse_one),
        ("second", pulse_one),
        ("first", other),
        ("second", other),
        ("first", pulse_two),
        ("second", pulse_two),
    ]


def test_scope_start_and_end_share_one_operation_count(tmp_path: Path) -> None:
    order: list[tuple[str, Occurrence[Any]]] = []
    recorder = _Recorder("recorder", order)
    context = _context(tmp_path, callbacks=[recorder])
    operation = _Work("first")

    with context.scope(operation) as started:
        assert started.event == Started(operation)
        assert started.count == 1
    with context.scope(_Work("second")) as second_started:
        assert second_started.count == 2

    assert [type(item.event) for item in recorder.occurrences] == [Started, Ended, Started, Ended]
    assert [item.count for item in recorder.occurrences] == [1, 1, 2, 2]
    assert recorder.occurrences[0].event.operation is operation
    assert recorder.occurrences[1].event.operation is operation


@pytest.mark.parametrize("lifecycle_type", [Started, Ended])
def test_emit_rejects_scope_managed_lifecycle_events(
    tmp_path: Path, lifecycle_type: type[Started[Any]] | type[Ended[Any]]
) -> None:
    context = _context(tmp_path)

    with pytest.raises(TypeError, match="scope"):
        context.emit(lifecycle_type(_Work("guarded")))

    assert not context.path("occurrences.jsonl").exists()


def test_emit_and_scope_reject_untyped_arguments(tmp_path: Path) -> None:
    context = _context(tmp_path)

    with suppress_type_checks():
        with pytest.raises(TypeError, match="Event"):
            context.emit(object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Operation"):
            with context.scope(object()):  # type: ignore[arg-type]
                pass


def test_scope_emits_end_when_the_body_raises(tmp_path: Path) -> None:
    order: list[tuple[str, Occurrence[Any]]] = []
    recorder = _Recorder("recorder", order)
    context = _context(tmp_path, callbacks=[recorder])

    with pytest.raises(RuntimeError, match="boom"):
        with context.scope(_Work("failing")):
            raise RuntimeError("boom")

    assert [type(item.event) for item in recorder.occurrences] == [Started, Ended]
    assert [item.count for item in recorder.occurrences] == [1, 1]


def test_scope_does_not_emit_end_when_started_dispatch_fails(tmp_path: Path) -> None:
    callback = _RaisesOnStarted()
    context = _context(tmp_path, callbacks=[callback])

    with pytest.raises(RuntimeError, match="started dispatch failed"):
        with context.scope(_Work("never-entered")):
            raise AssertionError("scope body must not run")

    assert [type(item.event) for item in callback.occurrences] == [Started]
    records = [
        json.loads(line)
        for line in context.path("occurrences.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == ["tpen.events.Started"]


def test_typed_occurrences_have_an_explicit_jsonl_edge_encoding(tmp_path: Path) -> None:
    context = _context(tmp_path)

    with context.scope(CollectSamples(step=7)):
        pass

    records = [
        json.loads(line)
        for line in context.path("occurrences.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    without_time = [
        {key: record[key] for key in record if key != "time"}
        for record in records
    ]
    assert without_time == [
        {
            "count": 1,
            "event": "tpen.events.Started",
            "fields": {"step": 7},
            "operation": "tpen.training.events.CollectSamples",
            "run_id": "typed-events-unit",
        },
        {
            "count": 1,
            "event": "tpen.events.Ended",
            "fields": {"step": 7},
            "operation": "tpen.training.events.CollectSamples",
            "run_id": "typed-events-unit",
        },
    ]
    assert all("step" not in record and "payload" not in record for record in records)


def test_typed_serialization_encodes_markers_and_public_dataclass_fields(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)

    context.emit(_MarkerEvent())
    with context.scope(_MarkerOperation()):
        pass
    context.emit(_PrivatePulse(label="visible", _private="hidden"))

    records = [
        json.loads(line)
        for line in context.path("occurrences.jsonl").read_text().splitlines()
    ]
    assert [record["fields"] for record in records] == [
        {},
        {},
        {},
        {"label": "visible"},
    ]


def test_typed_completion_serializes_nested_iteration_fields(tmp_path: Path) -> None:
    context = _context(tmp_path)
    iteration = TrainingIteration(step=7)

    context.emit(TrainingIterationCompleted(iteration=iteration))
    context.emit(UpdateCompleted(iteration=iteration))
    context.emit(UpdateSkipped(iteration=iteration))

    records = [
        json.loads(line)
        for line in context.path("occurrences.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "tpen.training.events.TrainingIterationCompleted",
        "tpen.training.events.UpdateCompleted",
        "tpen.training.events.UpdateSkipped",
    ]
    # Each concrete event type counts independently from one.
    assert [record["count"] for record in records] == [1, 1, 1]
    assert all(record["fields"] == {"iteration": {"step": 7}} for record in records)


def test_typed_event_serializes_a_plain_dataclass_field_field_wise(tmp_path: Path) -> None:
    """A domain object that is neither ``Event`` nor ``Operation`` must not collapse.

    ``RestoreReport`` is the concrete case D1 depends on: ``completed_updates``
    exists so a restored run's applied-update cursor reaches the durable record,
    and ``checkpoint_restored`` has no callback handlers, so the record is its
    only consumer.
    """

    context = _context(tmp_path)

    context.emit(
        _ReportPulse(
            report=RestoreReport(
                mode="train_resume",
                checkpoint_dir="checkpoints/step_000010",
                schema_version=2,
                next_iteration=11,
                completed_updates=10,
                loaded_model=True,
                loaded_optimizer=True,
                loaded_trainer=True,
                loaded_sampler=True,
                loaded_rng=True,
            )
        )
    )

    records = [
        json.loads(line)
        for line in context.path("occurrences.jsonl").read_text().splitlines()
    ]
    assert records[0]["fields"] == {
        "report": {
            "mode": "train_resume",
            "checkpoint_dir": "checkpoints/step_000010",
            "schema_version": 2,
            "next_iteration": 11,
            "completed_updates": 10,
            "loaded_model": True,
            "loaded_optimizer": True,
            "loaded_trainer": True,
            "loaded_sampler": True,
            "loaded_rng": True,
        }
    }


def test_typed_event_recurses_through_nested_plain_dataclasses(tmp_path: Path) -> None:
    """Nesting depth is unbounded, and private field names stay excluded."""

    context = _context(tmp_path)

    context.emit(
        _WrappedPulse(wrapper=_Wrapper(detail=_Detail(code=3, note="deep"), label="outer"))
    )

    records = [
        json.loads(line)
        for line in context.path("occurrences.jsonl").read_text().splitlines()
    ]
    assert records[0]["fields"] == {
        "wrapper": {"detail": {"code": 3, "note": "deep"}, "label": "outer"}
    }


def test_typed_event_does_not_recurse_into_containers(tmp_path: Path) -> None:
    """Pin the boundary: a dataclass behind a container keeps its type marker.

    No typed event carries a container of dataclasses, so ADR-E003 defers
    widening the traversal until one does. This test exists so that widening is
    a deliberate edit rather than an accident.
    """

    context = _context(tmp_path)

    context.emit(
        _ContainerPulse(
            details=(_Detail(code=1, note="first"),),
            by_name={"second": _Detail(code=2, note="second")},
        )
    )

    records = [
        json.loads(line)
        for line in context.path("occurrences.jsonl").read_text().splitlines()
    ]
    marker = {"type": f"{_Detail.__module__}.{_Detail.__name__}"}
    assert records[0]["fields"] == {"details": [marker], "by_name": {"second": marker}}


def test_typed_event_keeps_the_tensor_encoding_for_non_dataclass_values(
    tmp_path: Path,
) -> None:
    """A tensor is not a dataclass, so it keeps its shape/dtype/device encoding."""

    context = _context(tmp_path)

    context.emit(_TensorPulse(tensor=torch.zeros((2, 3), dtype=torch.float64)))

    records = [
        json.loads(line)
        for line in context.path("occurrences.jsonl").read_text().splitlines()
    ]
    assert records[0]["fields"] == {
        "tensor": {
            "device": "cpu",
            "dtype": "torch.float64",
            "shape": [2, 3],
            "type": "torch.Tensor",
        }
    }


def test_typed_event_rejects_a_cyclic_dataclass_field(tmp_path: Path) -> None:
    """Field-wise recursion refuses a cycle rather than exhausting the stack."""

    context = _context(tmp_path)
    node = _Cyclic()
    node.child = node

    with pytest.raises(TypeError, match="cyclic typed value"):
        context.emit(_CyclicPulse(node=node))

    assert not context.path("occurrences.jsonl").exists()


# The cycle guard tracks the instances open on the *current field path*, not
# every instance ever visited. That distinction is load-bearing and invisible:
# swapping the ancestors tuple for an all-visited set would make legitimate
# sharing raise "cyclic typed value", and before the two sharing tests below
# existed, every other test in this file still passed under that change. The
# sibling and diamond tests are the ones that fail on an all-visited set; the
# deep-chain test covers the other way the guard can go wrong, a depth budget
# standing in for a path scope.
#
# ``_field_paths`` turns a serialised record into a set of dotted leaf paths, so
# a *dropped* shared value surfaces as a missing path rather than as an
# assertion that happened not to look at it. Containers stay a pinned boundary
# -- the walker does not descend a list or a dict either, matching
# ``test_typed_event_does_not_recurse_into_containers``, so the walker and the
# code under test agree on scope by construction.


def _field_paths(value: object, prefix: str = "") -> list[str]:
    """Return every dotted leaf path under ``value``, tagged with its kind."""

    if isinstance(value, dict):
        if not value:
            return [f"{prefix}={{}}"]
        paths: list[str] = []
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else key
            paths.extend(_field_paths(value[key], child))
        return paths
    if isinstance(value, list):
        kinds = sorted({type(item).__name__ for item in value})
        return [f"{prefix}=[{len(value)}:{','.join(kinds)}]"]
    return [f"{prefix}:{type(value).__name__}"]


def _type_only_paths(value: object, prefix: str = "") -> list[str]:
    """Return the paths whose value collapsed to a bare ``{"type": ...}``."""

    if not isinstance(value, dict):
        return []
    if list(value) == ["type"]:
        return [f"{prefix}={value['type']}"]
    found: list[str] = []
    for key in sorted(value):
        child = f"{prefix}.{key}" if prefix else key
        found.extend(_type_only_paths(value[key], child))
    return found


def _emitted_fields(context: RunContext) -> dict[str, Any]:
    """Return the ``fields`` mapping of the single occurrence record written."""

    records = [
        json.loads(line)
        for line in context.path("occurrences.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1, records
    return records[0]["fields"]


def test_typed_event_serializes_one_instance_shared_by_two_sibling_fields(
    tmp_path: Path,
) -> None:
    """Sibling sharing is not a cycle: both occurrences must expand field-wise."""

    context = _context(tmp_path)
    shared = _Detail(code=7, note="shared")

    context.emit(_SiblingPulse(left=shared, right=shared))

    fields = _emitted_fields(context)
    expanded = {"code": 7, "note": "shared"}
    # Assert *both* paths: an implementation that expanded one occurrence and
    # dropped the other would satisfy a single-sided assertion.
    assert fields == {"left": expanded, "right": expanded}
    assert fields["left"] == fields["right"]
    assert set(_field_paths(fields)) == {
        "left.code:int",
        "left.note:str",
        "right.code:int",
        "right.note:str",
    }
    assert _type_only_paths(fields) == []


def test_typed_event_serializes_one_instance_shared_at_two_depths(
    tmp_path: Path,
) -> None:
    """Diamond sharing is not a cycle: depth 1 and depth 2 must both expand."""

    context = _context(tmp_path)
    shared = _Detail(code=7, note="shared")

    context.emit(
        _DiamondPulse(wrapper=_Wrapper(detail=shared, label="via-wrapper"), detail=shared)
    )

    fields = _emitted_fields(context)
    expanded = {"code": 7, "note": "shared"}
    assert fields == {
        "wrapper": {"detail": expanded, "label": "via-wrapper"},
        "detail": expanded,
    }
    assert fields["detail"] == fields["wrapper"]["detail"]
    assert set(_field_paths(fields)) == {
        "detail.code:int",
        "detail.note:str",
        "wrapper.detail.code:int",
        "wrapper.detail.note:str",
        "wrapper.label:str",
    }
    assert _type_only_paths(fields) == []


def test_typed_event_serializes_a_deep_acyclic_dataclass_chain(tmp_path: Path) -> None:
    """The guard is scoped to a field path, not to a depth budget.

    Depth 60 is the depth the recursion was measured clean at while the guard
    was written, and it is far past any plausible small depth cap. It stays
    cheap: the walk costs a small constant number of interpreter frames per
    level -- under 250 in total, comfortably inside CPython's default
    1000-frame limit even on top of pytest's own stack.
    """

    context = _context(tmp_path)
    depth = 60
    link = _Link(depth=depth - 1)
    for index in range(depth - 2, -1, -1):
        link = _Link(depth=index, child=link)

    context.emit(_ChainPulse(link=link))

    fields = _emitted_fields(context)
    expected = {f"link{'.child' * level}.depth:int" for level in range(depth)}
    expected.add(f"link{'.child' * depth}:NoneType")
    assert set(_field_paths(fields)) == expected
    assert _type_only_paths(fields) == []

    # Every level carries its own distinct payload, so no level was skipped,
    # truncated, or aliased onto another.
    depths = []
    node = fields["link"]
    while node is not None:
        depths.append(node["depth"])
        node = node["child"]
    assert depths == list(range(depth))


def test_training_phase_base_is_abstract() -> None:
    """A phase carrying no durable metric fragment cannot be constructed."""

    with pytest.raises(TypeError, match="phase_name"):
        TrainingPhase(step=0)


def test_typed_phase_serializes_only_its_step_field(tmp_path: Path) -> None:
    """``phase_name`` is a ClassVar, so it never reaches the occurrence edge."""

    context = _context(tmp_path)

    with context.scope(Backward(step=5)):
        pass

    records = [
        json.loads(line)
        for line in context.path("occurrences.jsonl").read_text().splitlines()
    ]
    assert [record["operation"] for record in records] == [
        "tpen.training.events.Backward",
        "tpen.training.events.Backward",
    ]
    assert all(record["fields"] == {"step": 5} for record in records)


@pytest.mark.parametrize(
    "value",
    [_StatefulEvent(), _StatefulOperation(), _SlottedStatefulEvent()],
)
def test_typed_serialization_rejects_stateful_non_dataclass_values(
    tmp_path: Path, value: Event | Operation
) -> None:
    context = _context(tmp_path)

    with pytest.raises(TypeError, match="must be a dataclass"):
        if isinstance(value, Event):
            context.emit(value)
        else:
            with context.scope(value):
                pass

    assert not context.path("occurrences.jsonl").exists()


def _complete_timed_iteration(context: RunContext, step: int) -> None:
    iteration = TrainingIteration(step=step)
    with context.scope(iteration):
        with context.scope(CollectSamples(step=step)):
            pass
        context.emit(TrainingIterationCompleted(iteration=iteration))


def test_train_phase_timing_reports_successful_typed_iteration(tmp_path: Path) -> None:
    logger = _Logger()
    callback = TrainPhaseTiming(clock=_Clock([1.0, 1.25]))
    context = _context(tmp_path, callbacks=[callback], loggers=[logger])

    _complete_timed_iteration(context, 3)

    assert len(logger.records) == 1
    assert logger.records[0].namespace == "train/perf"
    assert logger.records[0].step == 3
    assert logger.records[0].metrics == {"sampling_time_sec": 0.25}
    assert callback._phase_starts == {}
    assert callback._durations == {}


def test_train_phase_timing_converts_zero_based_start_to_occurrence_cadence(
    tmp_path: Path,
) -> None:
    logger = _Logger()
    callback = TrainPhaseTiming(
        every_n_steps=2,
        start_step=0,
        clock=_Clock([0.0, 0.1, 1.0, 1.1, 2.0, 2.2]),
    )
    context = _context(tmp_path, callbacks=[callback], loggers=[logger])

    _complete_timed_iteration(context, 0)
    _complete_timed_iteration(context, 1)
    _complete_timed_iteration(context, 2)

    assert [record.step for record in logger.records] == [0, 2]
    assert logger.records[0].metrics == {"sampling_time_sec": pytest.approx(0.1)}
    assert logger.records[1].metrics == {"sampling_time_sec": pytest.approx(0.2)}
    assert callback._phase_starts == {}
    assert callback._durations == {}


def test_train_phase_timing_none_interval_reports_every_success(
    tmp_path: Path,
) -> None:
    logger = _Logger()
    callback = TrainPhaseTiming(
        every_n_steps=None,
        clock=_Clock([0.0, 0.1, 1.0, 1.2]),
    )
    context = _context(tmp_path, callbacks=[callback], loggers=[logger])

    _complete_timed_iteration(context, 0)
    _complete_timed_iteration(context, 1)

    assert [record.step for record in logger.records] == [0, 1]


def test_train_phase_timing_failed_body_cleans_up_without_reporting(
    tmp_path: Path,
) -> None:
    logger = _Logger()
    callback = TrainPhaseTiming(clock=_Clock([1.0, 1.25, 2.0, 2.5]))
    context = _context(tmp_path, callbacks=[callback], loggers=[logger])
    iteration = TrainingIteration(step=3)

    with pytest.raises(RuntimeError, match="training failed"):
        with context.scope(iteration):
            with context.scope(CollectSamples(step=3)):
                pass
            with context.scope(Backward(step=3)):
                raise RuntimeError("training failed")

    assert logger.records == []
    assert callback._phase_starts == {}
    assert callback._durations == {}


def test_train_phase_timing_cleanup_does_not_mask_reporting_error(
    tmp_path: Path,
) -> None:
    callback = TrainPhaseTiming(clock=_Clock([1.0, 1.25]))
    context = _context(tmp_path, callbacks=[callback], loggers=[_RaisingLogger()])
    iteration = TrainingIteration(step=4)

    with pytest.raises(RuntimeError, match="timing report failed"):
        with context.scope(iteration):
            with context.scope(CollectSamples(step=4)):
                pass
            context.emit(TrainingIterationCompleted(iteration=iteration))

    assert callback._phase_starts == {}
    assert callback._durations == {}


def test_train_phase_timing_context_identity_change_clears_all_caches(
    tmp_path: Path,
) -> None:
    callback = TrainPhaseTiming(clock=_Clock([1.0, 2.0]))
    first = _context(tmp_path / "first", callbacks=[callback])
    second = _context(tmp_path / "second", callbacks=[callback])

    # Measure a phase in the first context without ever ending its iteration.
    with first.scope(Backward(step=8)):
        pass
    assert callback._durations
    with second.scope(TrainingIteration(step=0)):
        pass

    assert callback._phase_starts == {}
    assert callback._durations == {}


def test_train_phase_timing_rejects_duplicate_sampling_duration(tmp_path: Path) -> None:
    callback = TrainPhaseTiming(clock=_Clock([1.0, 1.25, 2.0, 2.5]))
    context = _context(tmp_path, callbacks=[callback])
    iteration = TrainingIteration(step=3)

    with pytest.raises(RuntimeError, match="duplicate sampling_time_sec"):
        with context.scope(iteration):
            with context.scope(CollectSamples(step=3)):
                pass
            with context.scope(CollectSamples(step=3)):
                pass

    assert callback._durations == {}
    assert callback._phase_starts == {}


def _context(
    tmp_path: Path,
    *,
    callbacks: list[Any] | None = None,
    loggers: list[Any] | None = None,
) -> RunContext:
    artifact_manager = ArtifactManager(
        tmp_path,
        experiment="typed-events",
        sector="unit",
        run_id="typed-events-unit",
        layout="flat",
    )
    artifact_manager.make_dirs()
    cfg = OmegaConf.create({})
    metadata = RunMetadata(
        run_id="typed-events-unit",
        run_name="typed-events-unit",
        timestamp="2026-08-05T12:00:00+00:00",
        timezone="UTC",
        git_commit="test-sha",
        git_branch="test-branch",
        dirty_worktree=False,
        command="pytest",
        config_path="test.yaml",
        resolved_config_path=str(artifact_manager.path("resolved_config.yaml")),
        run_dir=str(artifact_manager.run_dir),
        device="cpu",
        dtype="float64",
    )
    return RunContext(
        cfg=cfg,
        source_cfg=OmegaConf.create({}),
        artifact_manager=artifact_manager,
        metadata=metadata,
        clock=RunClock(timezone="UTC", tzinfo=UTC),
        callbacks=[] if callbacks is None else callbacks,
        loggers=[] if loggers is None else loggers,
    )
