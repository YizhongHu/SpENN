"""Focused tests for the typed-event foundation and sampling pilot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf
from typeguard import suppress_type_checks

from tpen.artifacts import ArtifactManager, RunClock, RunContext, RunMetadata
from tpen.callback import TrainPhaseTiming
from tpen.events import Ended, Event, Occurrence, Operation, Started
from tpen.training.events import CollectSamples


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

    def handle(self, event: object) -> None:
        del event

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


class _Clock:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        return self.values.pop(0)


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


def test_train_phase_timing_consumes_typed_collect_samples_scope(tmp_path: Path) -> None:
    logger = _Logger()
    callback = TrainPhaseTiming(clock=_Clock([1.0, 1.25]))
    context = _context(tmp_path, callbacks=[callback], loggers=[logger])

    with context.scope(CollectSamples(step=3)):
        pass
    context.emit_event("step_end", payload={"step": 3})

    assert len(logger.records) == 1
    assert logger.records[0].namespace == "train/perf"
    assert logger.records[0].step == 3
    assert logger.records[0].metrics == {"sampling_time_sec": 0.25}


def test_train_phase_timing_prunes_steps_skipped_by_legacy_cadence(tmp_path: Path) -> None:
    logger = _Logger()
    callback = TrainPhaseTiming(
        every_n_steps=2,
        clock=_Clock([0.0, 0.1, 1.0, 1.1, 2.0, 2.2]),
    )
    context = _context(tmp_path, callbacks=[callback], loggers=[logger])

    with context.scope(CollectSamples(step=0)):
        pass
    context.emit_event("step_end", payload={"step": 0})
    with context.scope(CollectSamples(step=1)):
        pass
    context.emit_event("step_end", payload={"step": 1})
    assert callback._durations[1]["sampling_time_sec"] == pytest.approx(0.1)
    callback._collect_samples_starts[99] = (1, -1.0)

    with context.scope(CollectSamples(step=2)):
        assert 1 not in callback._durations
        assert 99 not in callback._collect_samples_starts
        assert len(callback._collect_samples_starts) == 1
    context.emit_event("step_end", payload={"step": 2})

    assert [record.step for record in logger.records] == [0, 2]
    assert logger.records[0].metrics == {"sampling_time_sec": pytest.approx(0.1)}
    assert logger.records[1].metrics == {"sampling_time_sec": pytest.approx(0.2)}
    assert callback._collect_samples_starts == {}
    assert callback._durations == {}


def test_train_phase_timing_rejects_duplicate_sampling_duration(tmp_path: Path) -> None:
    callback = TrainPhaseTiming(clock=_Clock([1.0, 1.25, 2.0, 2.5]))
    context = _context(tmp_path, callbacks=[callback])

    with context.scope(CollectSamples(step=3)):
        pass
    with pytest.raises(RuntimeError, match="duplicate sampling_time_sec"):
        with context.scope(CollectSamples(step=3)):
            pass

    assert callback._durations[3] == {"sampling_time_sec": 0.25}
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
