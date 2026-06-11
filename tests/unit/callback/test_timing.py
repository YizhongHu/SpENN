"""Tests for runtime timing callbacks."""

from __future__ import annotations

import logging

import pytest

import spenn.callback as callback_module
from spenn.callback import DiagnosticTiming, EvaluationTiming, Event, RunTiming, Status, TrainStepTiming
from tests.unit.callback.support import FakeState, RecordingContext


class FakeClock:
    """Deterministic callable clock for timing tests."""

    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        if not self.values:
            raise AssertionError("fake clock exhausted")
        return self.values.pop(0)


def test_run_timing_logs_start_end_and_wall_time() -> None:
    context = RecordingContext()
    callback = RunTiming(clock=FakeClock([10.0, 12.5]), wall_clock=FakeClock([100.0, 103.0]))

    callback.handle(Event(name="run_start", context=context))
    callback.handle(Event(name="run_end", context=context))

    assert context.records == [
        {"metrics": {"start_time_unix": 100.0}, "step": 0, "namespace": "runtime", "event": None},
        {
            "metrics": {"end_time_unix": 103.0, "wall_time_sec": 2.5},
            "step": 0,
            "namespace": "runtime",
            "event": None,
        },
    ]


def test_run_timing_logs_failure_without_swallowing_exception() -> None:
    context = RecordingContext()
    callback = RunTiming(clock=FakeClock([1.0, 4.0]), wall_clock=FakeClock([10.0, 13.0]))

    callback.handle(Event(name="run_start", context=context))
    callback.handle(Event(name="exception", context=context, payload={"exception": RuntimeError("boom")}))

    assert context.records[-1]["metrics"] == {
        "end_time_unix": 13.0,
        "wall_time_sec": 3.0,
        "failed": True,
    }


def test_train_step_timing_logs_duration_and_rolling_mean() -> None:
    context = RecordingContext()
    callback = TrainStepTiming(rolling_window=2, clock=FakeClock([1.0, 1.5, 3.0, 4.0]))

    callback.handle(Event(name="step_start", context=context, payload={"step": 1}))
    callback.handle(Event(name="step_end", context=context, payload={"step": 1}))
    callback.handle(Event(name="step_start", context=context, payload={"step": 2}))
    callback.handle(Event(name="step_end", context=context, payload={"step": 2}))

    assert context.by_namespace("train/perf") == [
        {
            "metrics": {"step_time_sec": 0.5, "step_time_sec_rolling_mean": 0.5},
            "step": 1,
            "namespace": "train/perf",
            "event": None,
        },
        {
            "metrics": {"step_time_sec": 1.0, "step_time_sec_rolling_mean": 0.75},
            "step": 2,
            "namespace": "train/perf",
            "event": None,
        },
    ]


def test_status_can_render_train_step_timing_metric(caplog: pytest.LogCaptureFixture) -> None:
    context = RecordingContext()
    timing = TrainStepTiming(clock=FakeClock([1.0, 1.25]))
    status = Status(["step_end"], include=["train/perf/step_time_sec"], color="never")
    end_event = Event(
        name="step_end",
        context=context,
        state=FakeState(step=1),
        payload={"step": 1},
    )

    timing.handle(Event(name="step_start", context=context, payload={"step": 1}))
    timing.handle(end_event)
    with caplog.at_level(logging.INFO, logger="spenn.status"):
        status.handle(end_event)

    assert caplog.records[-1].getMessage() == "[train] step=1 step_time=0.25"


def test_evaluation_timing_logs_eval_perf_wall_time() -> None:
    context = RecordingContext()
    callback = EvaluationTiming(clock=FakeClock([2.0, 5.5]))

    callback.handle(Event(name="evaluate_start", context=context))
    callback.handle(Event(name="evaluate_end", context=context))

    assert context.latest("eval/perf") == {"wall_time_sec": 3.5}
    assert context.by_namespace("eval/perf")[-1]["step"] == 0


def test_diagnostic_timing_logs_named_diagnostic_duration() -> None:
    context = RecordingContext()
    callback = DiagnosticTiming(clock=FakeClock([7.0, 8.25]))

    callback.handle(
        Event(name="diagnostic_start", context=context, payload={"step": 4, "diagnostic_name": "energy"})
    )
    callback.handle(
        Event(name="diagnostic_end", context=context, payload={"step": 4, "diagnostic_name": "energy"})
    )

    assert context.latest("diagnostics/energy") == {"time_sec": 1.25}
    assert context.by_namespace("diagnostics/energy")[-1]["step"] == 4


def test_diagnostic_timing_requires_name() -> None:
    with pytest.raises(ValueError, match="diagnostic_name"):
        DiagnosticTiming(clock=FakeClock([1.0])).handle(
            Event(name="diagnostic_start", context=RecordingContext(), payload={"step": 1})
        )


def test_cuda_synchronize_flag_controls_cuda_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(callback_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(callback_module.torch.cuda, "synchronize", lambda: calls.append("sync"))

    no_sync = TrainStepTiming(cuda_synchronize=False, clock=FakeClock([1.0, 2.0]))
    no_sync.handle(Event(name="step_start", context=RecordingContext(), payload={"step": 1}))
    no_sync.handle(Event(name="step_end", context=RecordingContext(), payload={"step": 1}))
    assert calls == []

    with_sync = TrainStepTiming(cuda_synchronize=True, clock=FakeClock([1.0, 2.0]))
    with_sync.handle(Event(name="step_start", context=RecordingContext(), payload={"step": 1}))
    with_sync.handle(Event(name="step_end", context=RecordingContext(), payload={"step": 1}))
    assert calls == ["sync", "sync"]
