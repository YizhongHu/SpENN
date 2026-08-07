"""Focused tests for the typed-event foundation and sampling pilot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf
from typeguard import suppress_type_checks

from tpen.artifacts import (
    ArtifactManager,
    RunClock,
    RunContext,
    RunMetadata,
    write_event_artifact,
)
from tpen.callback import Event as LegacyEvent
from tpen.callback import TrainPhaseTiming
from tpen.events import Ended, Event, Occurrence, Operation, Started, Subscription
from tpen.events import ended, started
from tpen.runner.base import Runner
from tpen.training.events import (
    CollectSamples,
    TrainingIteration,
    TrainingIterationCompleted,
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


class _Recorder:
    def __init__(self, name: str, order: list[tuple[str, Occurrence[Any]]]) -> None:
        self.name = name
        self.order = order
        self.occurrences: list[Occurrence[Any]] = []
        self.legacy_events: list[object] = []

    def handle(self, event: object) -> None:
        self.legacy_events.append(event)

    def handle_occurrence(self, occurrence: Occurrence[Any], context: RunContext) -> None:
        del context
        self.occurrences.append(occurrence)
        self.order.append((self.name, occurrence))


class _RaisesOnStarted:
    def __init__(self) -> None:
        self.occurrences: list[Occurrence[Any]] = []

    def handle(self, event: object) -> None:
        del event

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


def test_legacy_and_typed_records_use_separate_jsonl_edges(tmp_path: Path) -> None:
    context = _context(tmp_path)

    context.emit_event("legacy_event", payload={"step": 4, "value": "legacy"})
    context.emit(_Pulse("typed"))

    legacy_records = [
        json.loads(line) for line in context.path("events.jsonl").read_text().splitlines()
    ]
    typed_records = [
        json.loads(line)
        for line in context.path("occurrences.jsonl").read_text().splitlines()
    ]
    without_time = [
        {key: value for key, value in record.items() if key != "time"}
        for record in legacy_records
    ]
    assert without_time == [
        {
            "event": "legacy_event",
            "payload": {"step": 4, "value": "legacy"},
            "run_id": "typed-events-unit",
            "step": 4,
        }
    ]
    assert len(typed_records) == 1
    assert typed_records[0]["count"] == 1
    assert typed_records[0]["fields"] == {"label": "typed"}
    assert "payload" not in typed_records[0]
    assert "count" not in legacy_records[0]


def test_direct_legacy_event_payload_does_not_populate_explicit_step() -> None:
    event = LegacyEvent(
        name="direct",
        context=None,  # type: ignore[arg-type]
        state=SimpleNamespace(global_step=99),
        payload={"step": 4},
    )

    assert event.step is None


def test_legacy_ingress_normalizes_steps_and_preserves_jsonl_shapes(
    tmp_path: Path,
) -> None:
    recorder = _Recorder("legacy", [])
    context = _context(tmp_path, callbacks=[recorder])

    context.emit_event("payload_only", payload={"step": 1, "value": "legacy"})
    explicit_payload: dict[str, Any] = {"value": "explicit"}
    context.emit_event(
        "explicit_only",
        state=SimpleNamespace(global_step=99),
        payload=explicit_payload,
        step=2,
    )
    context.emit_event("matching_dual", payload={"step": "3"}, step=3)
    context.emit_event(
        "stepless",
        state=SimpleNamespace(global_step=123),
        payload={},
    )

    delivered = recorder.legacy_events
    assert [event.step for event in delivered] == [1, 2, 3, None]
    assert explicit_payload == {"value": "explicit"}
    records = [
        json.loads(line)
        for line in context.path("events.jsonl").read_text().splitlines()
    ]
    without_time = [
        {key: value for key, value in record.items() if key != "time"}
        for record in records
    ]
    assert without_time == [
        {
            "event": "payload_only",
            "payload": {"step": 1, "value": "legacy"},
            "run_id": "typed-events-unit",
            "step": 1,
        },
        {
            "event": "explicit_only",
            "payload": {"step": 2, "value": "explicit"},
            "run_id": "typed-events-unit",
            "step": 2,
        },
        {
            "event": "matching_dual",
            "payload": {"step": "3"},
            "run_id": "typed-events-unit",
            "step": 3,
        },
        {
            "event": "stepless",
            "payload": {},
            "run_id": "typed-events-unit",
            "step": None,
        },
    ]


def test_legacy_ingress_step_mismatch_writes_and_dispatches_nothing(
    tmp_path: Path,
) -> None:
    recorder = _Recorder("legacy", [])
    context = _context(tmp_path, callbacks=[recorder])

    with pytest.raises(ValueError, match="step mismatch"):
        context.emit_event("mismatch", payload={"step": 2}, step=1)

    assert recorder.legacy_events == []
    assert not context.path("events.jsonl").exists()


def test_event_artifact_rejects_direct_step_mismatch_before_append(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    event = LegacyEvent(
        name="mismatch",
        context=context,
        payload={"step": 2},
        step=1,
    )

    with pytest.raises(ValueError, match="step mismatch"):
        write_event_artifact(context, event)

    assert not context.path("events.jsonl").exists()


def test_runner_fallback_uses_the_same_legacy_step_adapter() -> None:
    recorder = _Recorder("legacy", [])
    context = object.__new__(RunContext)
    context.callbacks = [recorder]
    runner = Runner()

    runner.emit("step_end", context, payload={"step": 5}, step=5)
    with pytest.raises(ValueError, match="step mismatch"):
        runner.emit("step_end", context, payload={"step": 6}, step=5)

    assert [event.step for event in recorder.legacy_events] == [5]


def test_migrated_and_compatibility_event_names_keep_legacy_payload_schema(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)

    context.emit_event(
        "train_phase_start",
        payload={"phase": "forward"},
        step=0,
    )
    context.emit_event("step_start", step=0)
    context.emit_event("step_end", step=0)
    context.emit_event("train_end", step=1)
    context.emit_event(
        "diagnostic_start",
        payload={"step": 4, "diagnostic_name": "energy"},
    )
    context.emit_event(
        "load_start",
        payload={"mode": "train_resume", "path": "checkpoint"},
    )
    context.emit_event("run_failed", payload={"phase": "run"})

    records = [
        json.loads(line)
        for line in context.path("events.jsonl").read_text().splitlines()
    ]
    by_name = {record["event"]: record for record in records}
    assert by_name["train_phase_start"]["payload"] == {
        "phase": "forward",
        "step": 0,
    }
    assert by_name["step_start"]["payload"] == {"step": 0}
    assert by_name["step_end"]["payload"] == {"step": 0}
    assert by_name["train_end"]["payload"] == {"step": 1}
    assert by_name["diagnostic_start"]["step"] == 4
    assert by_name["diagnostic_start"]["payload"]["step"] == 4
    assert by_name["load_start"]["step"] is None
    assert "step" not in by_name["load_start"]["payload"]
    assert by_name["run_failed"]["step"] is None
    assert "step" not in by_name["run_failed"]["payload"]


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

    context.emit(TrainingIterationCompleted(iteration=TrainingIteration(step=7)))

    record = json.loads(context.path("occurrences.jsonl").read_text().strip())
    assert record["event"] == "tpen.training.events.TrainingIterationCompleted"
    assert record["count"] == 1
    assert record["fields"] == {"iteration": {"step": 7}}


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
    assert callback._starts == {}
    assert callback._collect_samples_starts == {}
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
    assert callback._collect_samples_starts == {}
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
    callback = TrainPhaseTiming(clock=_Clock([1.0, 1.25, 2.0]))
    context = _context(tmp_path, callbacks=[callback], loggers=[logger])
    iteration = TrainingIteration(step=3)

    with pytest.raises(RuntimeError, match="training failed"):
        with context.scope(iteration):
            with context.scope(CollectSamples(step=3)):
                pass
            context.emit_event(
                "train_phase_start",
                payload={"step": 3, "phase": "backward"},
                step=3,
            )
            raise RuntimeError("training failed")

    assert logger.records == []
    assert callback._starts == {}
    assert callback._collect_samples_starts == {}
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

    assert callback._starts == {}
    assert callback._collect_samples_starts == {}
    assert callback._durations == {}


def test_train_phase_timing_context_identity_change_clears_all_caches(
    tmp_path: Path,
) -> None:
    callback = TrainPhaseTiming(clock=_Clock([1.0]))
    first = _context(tmp_path / "first", callbacks=[callback])
    second = _context(tmp_path / "second", callbacks=[callback])

    first.emit_event(
        "train_phase_start",
        payload={"step": 8, "phase": "backward"},
        step=8,
    )
    assert callback._starts
    with second.scope(TrainingIteration(step=0)):
        pass

    assert callback._starts == {}
    assert callback._collect_samples_starts == {}
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
    assert callback._collect_samples_starts == {}


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
