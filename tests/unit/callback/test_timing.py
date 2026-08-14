"""Tests for runtime timing callbacks."""

from __future__ import annotations

import inspect

import logging
from pathlib import Path

import pytest

from tpen.callback import (
    DiagnosticTiming,
    EvaluationComponentTiming,
    EvaluationTiming,
    RunTiming,
    Status,
    TrainPhaseTiming,
    TrainStepTiming,
)
from tpen.callback.timing import base as timing_base
from tpen.evaluation.events import (
    CalculatorRun,
    ComponentRun,
    EvaluationCompleted,
    EvaluationStarted,
    EvaluationTaskRun,
    GeneratorRun,
    SummaryRun,
)
from tpen.evaluation.results import TaskResult
from tpen.evaluation.state import EvaluationRunState
from tpen.events import Ended, Occurrence, Started
from tpen.run_events import RunCompleted, RunFailed, RunStarted
from tpen.training.events import (
    Backward,
    BuildBatch,
    CollectSamples,
    Forward,
    LocalEnergy,
    Metrics,
    Objective,
    OptimizerUpdate,
    TrainingIteration,
    TrainingIterationCompleted,
    TrainingPhase,
)
from tests.unit.callback.support import (
    RecordingContext,
    deliver_completed_iteration,
    training_state,
)


class FakeClock:
    """Deterministic callable clock for timing tests."""

    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        if not self.values:
            raise AssertionError("fake clock exhausted")
        return self.values.pop(0)


def _dispatch_iteration_start(
    callback: TrainPhaseTiming,
    context: RecordingContext,
    *,
    step: int,
    count: int,
) -> TrainingIteration:
    iteration = TrainingIteration(step=step)
    callback.handle_occurrence(
        Occurrence(event=Started(iteration), count=count),
        context,
    )
    return iteration


def _dispatch_phase_start(
    callback: TrainPhaseTiming,
    context: RecordingContext,
    phase: TrainingPhase,
    *,
    count: int,
) -> None:
    """Dispatch only the ``Started`` boundary of one typed phase scope."""

    callback.handle_occurrence(Occurrence(event=Started(phase), count=count), context)


def _dispatch_phase(
    callback: TrainPhaseTiming,
    context: RecordingContext,
    phase: TrainingPhase,
    *,
    count: int,
) -> None:
    """Dispatch one complete typed phase scope, Started then Ended."""

    _dispatch_phase_start(callback, context, phase, count=count)
    callback.handle_occurrence(Occurrence(event=Ended(phase), count=count), context)


def _dispatch_iteration_success(
    callback: TrainPhaseTiming,
    context: RecordingContext,
    *,
    iteration: TrainingIteration,
    count: int,
) -> None:
    callback.handle_occurrence(
        Occurrence(
            event=TrainingIterationCompleted(iteration=iteration),
            count=count,
        ),
        context,
    )
    callback.handle_occurrence(
        Occurrence(event=Ended(iteration), count=count),
        context,
    )


def _deliver_run_event(callback: object, context: RecordingContext, event: object) -> None:
    """Hand one `tpen.run_events` occurrence to a migrated run-level callback."""

    callback.handle_occurrence(Occurrence(event=event, count=1), context)


def test_run_timing_logs_start_end_and_wall_time() -> None:
    context = RecordingContext()
    callback = RunTiming(clock=FakeClock([10.0, 12.5]), wall_clock=FakeClock([100.0, 103.0]))

    _deliver_run_event(callback, context, RunStarted())
    _deliver_run_event(callback, context, RunCompleted())

    assert context.records == [
        {"metrics": {"start_time_unix": 100.0}, "step": 0, "namespace": "runtime"},
        {
            "metrics": {"end_time_unix": 103.0, "wall_time_sec": 2.5},
            "step": 0,
            "namespace": "runtime",
        },
    ]


def test_run_timing_logs_one_failed_record_per_failed_run() -> None:
    """`RunFailed` replaces the ``run_failed``/``exception`` pair this answered.

    Both strings carried the same payload, so a failed run logged ``runtime``
    twice -- with a later ``end_time_unix`` the second time, so not even
    identically. The clock holds exactly two values, which makes a regression to
    two firings raise rather than merely compare unequal.
    """

    context = RecordingContext()
    callback = RunTiming(clock=FakeClock([1.0, 4.0]), wall_clock=FakeClock([10.0, 13.0]))

    _deliver_run_event(callback, context, RunStarted())
    _deliver_run_event(callback, context, RunFailed(exception_type="RuntimeError", exception_message="boom"))

    assert len(context.by_namespace("runtime")) == 2
    assert context.records[-1]["metrics"] == {
        "end_time_unix": 13.0,
        "wall_time_sec": 3.0,
        "failed": True,
    }


def test_run_timing_marks_a_returned_failed_run_completed_boundary() -> None:
    context = RecordingContext()
    callback = RunTiming(clock=FakeClock([1.0, 4.0]), wall_clock=FakeClock([10.0, 13.0]))

    _deliver_run_event(callback, context, RunStarted())
    _deliver_run_event(callback, context, RunCompleted(status="failed"))

    assert context.latest("runtime") == {
        "end_time_unix": 13.0,
        "wall_time_sec": 3.0,
        "failed": True,
    }


def test_train_step_timing_logs_duration_and_rolling_mean() -> None:
    context = RecordingContext()
    callback = TrainStepTiming(rolling_window=2, clock=FakeClock([1.0, 1.5, 3.0, 4.0]))
    state = training_state()

    for count, step in ((1, 1), (2, 2)):
        callback.handle_occurrence(
            Occurrence(event=Started(TrainingIteration(step=step)), count=count), context, state
        )
        callback.handle_occurrence(
            Occurrence(event=Ended(TrainingIteration(step=step)), count=count), context, state
        )

    assert context.by_namespace("train/perf") == [
        {
            "metrics": {"step_time_sec": 0.5, "step_time_sec_rolling_mean": 0.5},
            "step": 1,
            "namespace": "train/perf",
        },
        {
            "metrics": {"step_time_sec": 1.0, "step_time_sec_rolling_mean": 0.75},
            "step": 2,
            "namespace": "train/perf",
        },
    ]
    assert state.timing is not None
    assert state.timing.step_time_sec_rolling_mean == 0.75



def test_train_step_timing_applies_legacy_step_cadence_to_typed_boundaries() -> None:
    """The paired lifecycle gate admits the same one-based cadence as legacy steps."""

    context = RecordingContext()
    callback = TrainStepTiming(
        every_n_steps=2, clock=FakeClock([1.0, 1.5])
    )
    state = training_state()

    for count, step in ((1, 1), (2, 2)):
        iteration = TrainingIteration(step=step)
        callback.handle_occurrence(Occurrence(event=Started(iteration), count=count), context, state)
        callback.handle_occurrence(Occurrence(event=Ended(iteration), count=count), context, state)

    assert context.by_namespace("train/perf") == [
        {
            "metrics": {"step_time_sec": 0.5, "step_time_sec_rolling_mean": 0.5},
            "step": 2,
            "namespace": "train/perf",
        }
    ]


def test_train_step_timing_discards_a_failed_iteration_boundary() -> None:
    """A partial iteration is not a completed-step performance sample."""

    context = RecordingContext()
    callback = TrainStepTiming(clock=FakeClock([1.0, 2.0]))
    state = training_state()
    iteration = TrainingIteration(step=1)

    callback.handle_occurrence(Occurrence(event=Started(iteration), count=1), context, state)
    callback.handle_occurrence(
        Occurrence(event=Ended(iteration, succeeded=False), count=1), context, state
    )

    assert context.by_namespace("train/perf") == []
    assert state.timing is None

def test_train_step_timing_feeds_status_line_through_typed_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Timing reaches `Status` through the typed `TrainerState` contract."""

    context = RecordingContext()
    timing = TrainStepTiming(clock=FakeClock([1.0, 1.25]))
    status = Status(include=["train/perf/step_time_sec"], color="never", train_lines=True)
    state = training_state(step=1)

    timing.handle_occurrence(
        Occurrence(event=Started(TrainingIteration(step=1)), count=1), context, state
    )
    timing.handle_occurrence(
        Occurrence(event=Ended(TrainingIteration(step=1)), count=1), context, state
    )
    with caplog.at_level(logging.INFO, logger="tpen.status"):
        deliver_completed_iteration(status, context, state, step=1)

    assert context.latest("train/perf")["step_time_sec"] == 0.25
    assert caplog.records[-1].getMessage() == "[train] step=1 step_time=0.25"


def test_train_phase_timing_logs_one_record_per_successful_completion() -> None:
    context = RecordingContext()
    callback = TrainPhaseTiming(clock=FakeClock([1.0, 1.25, 2.0, 2.75]))
    iteration = _dispatch_iteration_start(callback, context, step=3, count=1)

    _dispatch_phase(callback, context, BuildBatch(step=3), count=1)
    _dispatch_phase(callback, context, Backward(step=3), count=1)
    _dispatch_iteration_success(
        callback,
        context,
        iteration=iteration,
        count=1,
    )

    assert context.by_namespace("train/perf") == [
        {
            "metrics": {"batch_build_time_sec": 0.25, "backward_time_sec": 0.75},
            "step": 3,
            "namespace": "train/perf",
        }
    ]


def test_train_phase_timing_derives_every_metric_key_from_phase_name() -> None:
    """Each concrete phase type owns exactly one durable timing metric key."""

    context = RecordingContext()
    phases = (
        CollectSamples(step=0),
        BuildBatch(step=0),
        LocalEnergy(step=0),
        Forward(step=0),
        Objective(step=0),
        Backward(step=0),
        OptimizerUpdate(step=0),
        Metrics(step=0),
    )
    # Two exactly representable clock reads per phase, so every duration is 0.5.
    callback = TrainPhaseTiming(
        clock=FakeClock(
            [value for index in range(len(phases)) for value in (float(index), index + 0.5)]
        )
    )
    iteration = _dispatch_iteration_start(callback, context, step=0, count=1)

    for phase in phases:
        # Occurrence counts are per concrete operation type, so each phase
        # scope in one iteration is its own first occurrence.
        _dispatch_phase(callback, context, phase, count=1)
    _dispatch_iteration_success(callback, context, iteration=iteration, count=1)

    assert context.by_namespace("train/perf") == [
        {
            "metrics": {
                "sampling_time_sec": 0.5,
                "batch_build_time_sec": 0.5,
                "local_energy_time_sec": 0.5,
                "forward_time_sec": 0.5,
                "objective_time_sec": 0.5,
                "backward_time_sec": 0.5,
                "optimizer_step_time_sec": 0.5,
                "post_step_metrics_time_sec": 0.5,
            },
            "step": 0,
            "namespace": "train/perf",
        }
    ]


def test_train_phase_timing_completion_without_phases_logs_nothing() -> None:
    context = RecordingContext()
    callback = TrainPhaseTiming(clock=FakeClock([]))
    iteration = _dispatch_iteration_start(callback, context, step=1, count=1)

    _dispatch_iteration_success(
        callback,
        context,
        iteration=iteration,
        count=1,
    )

    assert context.records == []


def test_train_phase_timing_drops_unmatched_phase_starts_at_iteration_end() -> None:
    context = RecordingContext()
    callback = TrainPhaseTiming(clock=FakeClock([1.0, 5.0, 5.5]))

    # A phase started in step 1 but never finished must not leak into step 2.
    first = _dispatch_iteration_start(callback, context, step=1, count=1)
    _dispatch_phase_start(callback, context, BuildBatch(step=1), count=1)
    _dispatch_iteration_success(
        callback,
        context,
        iteration=first,
        count=1,
    )
    # Phase starts are keyed by ``(phase type, occurrence count)``, so a leaked
    # entry can never collide with a later phase and the reported records alone
    # cannot observe the leak. Assert the cleanup directly.
    assert callback._phase_starts == {}

    second = _dispatch_iteration_start(callback, context, step=2, count=2)
    _dispatch_phase(callback, context, BuildBatch(step=2), count=2)
    _dispatch_iteration_success(
        callback,
        context,
        iteration=second,
        count=2,
    )

    assert context.by_namespace("train/perf") == [
        {
            "metrics": {"batch_build_time_sec": 0.5},
            "step": 2,
            "namespace": "train/perf",
        }
    ]
    assert callback._phase_starts == {}


def _dispatch_component(
    callback: EvaluationComponentTiming,
    context: RecordingContext,
    operation: ComponentRun,
    *,
    count: int,
) -> None:
    """Dispatch one complete typed component scope, Started then Ended."""

    callback.handle_occurrence(Occurrence(event=Started(operation), count=count), context)
    callback.handle_occurrence(Occurrence(event=Ended(operation), count=count), context)


def _dispatch_task_start(
    callback: EvaluationComponentTiming,
    context: RecordingContext,
    task: EvaluationTaskRun,
    *,
    count: int,
) -> None:
    callback.handle_occurrence(Occurrence(event=Started(task), count=count), context)


def _dispatch_task_end(
    callback: EvaluationComponentTiming,
    context: RecordingContext,
    task: EvaluationTaskRun,
    *,
    count: int,
) -> None:
    callback.handle_occurrence(Occurrence(event=Ended(task), count=count), context)


def _task_run(name: str = "energy") -> EvaluationTaskRun:
    return EvaluationTaskRun(name=name, namespace=f"eval/{name}", output_dir=Path("/runs") / name)


def _task_result(name: str = "energy", *, status: str = "success") -> TaskResult:
    return TaskResult(
        name=name,
        namespace=f"eval/{name}",
        output_dir=Path("/runs") / name,
        status=status,
        metrics={},
        artifacts=(),
        failures=(),
    )


def test_evaluation_timing_logs_eval_perf_wall_time() -> None:
    context = RecordingContext()
    callback = EvaluationTiming(clock=FakeClock([2.0, 5.5]))

    callback.handle_occurrence(Occurrence(event=EvaluationStarted(), count=1), context)
    callback.handle_occurrence(Occurrence(event=EvaluationCompleted(), count=1), context)

    assert context.latest("eval/perf") == {"wall_time_sec": 3.5}
    assert context.by_namespace("eval/perf")[-1]["step"] == 0


def test_evaluation_timing_marks_a_failed_suite_completion() -> None:
    context = RecordingContext()
    callback = EvaluationTiming(clock=FakeClock([2.0, 5.5]))

    callback.handle_occurrence(Occurrence(event=EvaluationStarted(), count=1), context)
    callback.handle_occurrence(
        Occurrence(event=EvaluationCompleted(status="failed"), count=1), context
    )

    assert context.latest("eval/perf") == {"wall_time_sec": 3.5, "failed": True}


def test_evaluation_timing_reports_failed_off_the_typed_run_failure() -> None:
    """The only writer of ``eval/perf {failed: True}``, now fully typed.

    The evaluation domain has no suite-level failure moment to hang an event on,
    so this metric has always come from the RUN's failure boundary. That used to
    be the ``exception`` string, the last legacy run-level trigger in the
    codebase; it is now `tpen.run_events.RunFailed`, and the metric is unchanged
    (ADR-E006).
    """

    context = RecordingContext()
    callback = EvaluationTiming(clock=FakeClock([2.0, 6.0]))

    callback.handle_occurrence(Occurrence(event=EvaluationStarted(), count=1), context)
    _deliver_run_event(callback, context, RunFailed(exception_type="E", exception_message="m"))

    assert context.latest("eval/perf") == {"wall_time_sec": 4.0, "failed": True}
    assert context.by_namespace("eval/perf")[-1]["step"] == 0


def test_evaluation_timing_reports_nothing_if_evaluation_never_started() -> None:
    context = RecordingContext()
    callback = EvaluationTiming(clock=FakeClock([]))

    _deliver_run_event(callback, context, RunFailed(exception_type="E", exception_message="m"))

    assert context.records == []


def test_evaluation_component_timing_logs_one_record_per_task_at_the_task_scope_end() -> None:
    context = RecordingContext()
    callback = EvaluationComponentTiming(clock=FakeClock([1.0, 1.25, 2.0, 2.75, 3.0, 3.5]))
    task = _task_run()

    _dispatch_task_start(callback, context, task, count=1)
    _dispatch_component(callback, context, GeneratorRun(name="mcmc"), count=1)
    _dispatch_component(callback, context, CalculatorRun(name="local_energy"), count=1)
    _dispatch_component(callback, context, SummaryRun(name="energy_stats"), count=1)
    _dispatch_task_end(callback, context, task, count=1)

    assert context.by_namespace("eval/perf/energy") == [
        {
            "metrics": {
                # The generator key is FLAT by design; a task has exactly one.
                "generator_time_sec": 0.25,
                "calculator/local_energy_time_sec": 0.75,
                "summary/energy_stats_time_sec": 0.5,
            },
            "step": 0,
            "namespace": "eval/perf/energy",
        }
    ]


def test_evaluation_component_timing_derives_every_key_from_component_kind() -> None:
    """ADR-E006: no component-kind literal may survive inside the callback."""

    context = RecordingContext()
    callback = EvaluationComponentTiming(clock=FakeClock([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    task = _task_run()

    _dispatch_task_start(callback, context, task, count=1)
    for index, operation in enumerate(
        (GeneratorRun(name="g"), CalculatorRun(name="c"), SummaryRun(name="s")), start=1
    ):
        _dispatch_component(callback, context, operation, count=index)
    _dispatch_task_end(callback, context, task, count=1)

    assert set(context.latest("eval/perf/energy")) == {
        f"{GeneratorRun.component_kind}_time_sec",
        f"{CalculatorRun.component_kind}/c_time_sec",
        f"{SummaryRun.component_kind}/s_time_sec",
    }


def test_evaluation_component_timing_flushes_measured_components_of_a_failed_task() -> None:
    """A failed task still reports whatever was measured before it failed.

    The typed path needs no separate failure branch for this: the task scope's
    ``Ended`` fires from a ``finally``, so one boundary covers both outcomes
    where the legacy path needed ``task_end`` and ``task_failed``.
    """

    context = RecordingContext()
    callback = EvaluationComponentTiming(clock=FakeClock([1.0, 1.5]))
    task = _task_run()

    _dispatch_task_start(callback, context, task, count=1)
    _dispatch_component(callback, context, GeneratorRun(name="mcmc"), count=1)
    _dispatch_task_end(callback, context, task, count=1)

    assert context.by_namespace("eval/perf/energy") == [
        {
            "metrics": {"generator_time_sec": 0.5},
            "step": 0,
            "namespace": "eval/perf/energy",
        }
    ]


def test_evaluation_component_timing_task_without_components_logs_nothing() -> None:
    context = RecordingContext()
    callback = EvaluationComponentTiming(clock=FakeClock([]))
    task = _task_run()

    _dispatch_task_start(callback, context, task, count=1)
    _dispatch_task_end(callback, context, task, count=1)

    assert context.records == []


def test_evaluation_component_timing_accumulates_repeated_component_names() -> None:
    context = RecordingContext()
    callback = EvaluationComponentTiming(clock=FakeClock([0.0, 1.0, 10.0, 10.5]))
    task = _task_run()

    _dispatch_task_start(callback, context, task, count=1)
    _dispatch_component(callback, context, CalculatorRun(name="same"), count=1)
    _dispatch_component(callback, context, CalculatorRun(name="same"), count=2)
    _dispatch_task_end(callback, context, task, count=1)

    assert context.latest("eval/perf/energy") == {"calculator/same_time_sec": 1.5}


def test_evaluation_component_timing_drops_unmatched_starts_at_the_task_boundary() -> None:
    context = RecordingContext()
    callback = EvaluationComponentTiming(clock=FakeClock([1.0, 5.0, 5.5]))
    task = _task_run()

    # A component started in one task but never finished must not leak into the
    # next run of the same task.
    _dispatch_task_start(callback, context, task, count=1)
    callback.handle_occurrence(
        Occurrence(event=Started(CalculatorRun(name="local_energy")), count=1), context
    )
    _dispatch_task_end(callback, context, task, count=1)
    _dispatch_task_start(callback, context, task, count=2)
    _dispatch_component(callback, context, CalculatorRun(name="local_energy"), count=2)
    _dispatch_task_end(callback, context, task, count=2)

    assert context.by_namespace("eval/perf/energy") == [
        {
            "metrics": {"calculator/local_energy_time_sec": 0.5},
            "step": 0,
            "namespace": "eval/perf/energy",
        }
    ]
    assert callback._starts == {}


def test_evaluation_component_timing_requires_an_enclosing_task_scope() -> None:
    context = RecordingContext()
    callback = EvaluationComponentTiming(clock=FakeClock([1.0]))

    with pytest.raises(ValueError, match="EvaluationTaskRun"):
        callback.handle_occurrence(
            Occurrence(event=Started(GeneratorRun(name="mcmc")), count=1), context
        )


def test_evaluation_component_timing_requires_a_named_component() -> None:
    context = RecordingContext()
    callback = EvaluationComponentTiming(clock=FakeClock([1.0]))

    _dispatch_task_start(callback, context, _task_run(), count=1)
    with pytest.raises(ValueError, match="calculator"):
        callback.handle_occurrence(
            Occurrence(event=Started(CalculatorRun(name=None)), count=1), context
        )


def _dispatch_diagnostic_task(
    callback: DiagnosticTiming,
    context: RecordingContext,
    task: EvaluationTaskRun,
    state: EvaluationRunState,
    *,
    count: int = 1,
) -> None:
    callback.handle_occurrence(Occurrence(event=Started(task), count=count), context, state)
    callback.handle_occurrence(Occurrence(event=Ended(task), count=count), context, state)


def test_diagnostic_timing_logs_named_task_duration() -> None:
    context = RecordingContext()
    callback = DiagnosticTiming(clock=FakeClock([7.0, 8.25]))
    state = EvaluationRunState(task_result=_task_result())

    _dispatch_diagnostic_task(callback, context, _task_run(), state)

    assert context.latest("diagnostics/energy") == {"time_sec": 1.25}
    # Evaluation's coordinate is the task namespace, so every record is step 0.
    assert context.by_namespace("diagnostics/energy")[-1]["step"] == 0


@pytest.mark.parametrize("status", ["failed", "partial_failed"])
def test_diagnostic_timing_reports_failed_from_the_delivered_state(status: str) -> None:
    """The published ``failed`` flag is readable ONLY from the domain state.

    This is the evidence ADR-E008 asked D1 for: the legacy path discriminated
    the same moment by comparing a status string to choose between ``task_end``
    and ``task_failed``, and the typed path reads the status off the result the
    evaluator wrote into the state inside the scope body.
    """

    context = RecordingContext()
    callback = DiagnosticTiming(clock=FakeClock([0.0, 2.0]))
    state = EvaluationRunState(task_result=_task_result(status=status))

    _dispatch_diagnostic_task(callback, context, _task_run(), state)

    assert context.latest("diagnostics/energy") == {"time_sec": 2.0, "failed": True}


def test_diagnostic_timing_logs_nothing_when_the_task_body_raised() -> None:
    """``Ended`` fires from a ``finally``, so it is reached with no result at all.

    The legacy path emitted neither ``task_end`` nor ``task_failed`` there, so it
    wrote no ``diagnostics/`` record; publishing a duration with a guessed status
    instead would be a new, wrong metric.
    """

    context = RecordingContext()
    callback = DiagnosticTiming(clock=FakeClock([0.0]))
    state = EvaluationRunState()

    _dispatch_diagnostic_task(callback, context, _task_run(), state)

    assert context.records == []
    assert callback._starts == {}


@pytest.mark.parametrize(
    "callback_type",
    (
        DiagnosticTiming,
        EvaluationComponentTiming,
        EvaluationTiming,
        RunTiming,
        TrainPhaseTiming,
        TrainStepTiming,
    ),
)
def test_all_timing_callbacks_publish_only_accelerator_synchronize(callback_type: type) -> None:
    signature = inspect.signature(callback_type)
    assert "accelerator_synchronize" in signature.parameters
    assert "cuda_synchronize" not in signature.parameters

    callback_type(accelerator_synchronize=True)
    with pytest.raises(TypeError, match="cuda_synchronize"):
        callback_type(cuda_synchronize=True)


def test_accelerator_synchronize_flag_controls_device_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    # The public option is `accelerator_synchronize` as of v0.3.0 (formerly
    # `cuda_synchronize`); the work is delegated to the backend-agnostic accelerator
    # helper, so the seam under test is that helper rather than torch.cuda.
    calls: list[str] = []
    context = RecordingContext()
    monkeypatch.setattr(
        timing_base,
        "_synchronize_accelerator",
        lambda **kwargs: calls.append("sync"),
    )

    no_sync = TrainStepTiming(accelerator_synchronize=False, clock=FakeClock([1.0, 2.0]))
    context = RecordingContext()
    state = training_state()
    no_sync.handle_occurrence(
        Occurrence(event=Started(TrainingIteration(step=1)), count=1), context, state
    )
    no_sync.handle_occurrence(
        Occurrence(event=Ended(TrainingIteration(step=1)), count=1), context, state
    )
    assert calls == []

    with_sync = TrainStepTiming(accelerator_synchronize=True, clock=FakeClock([1.0, 2.0]))
    with_sync.handle_occurrence(
        Occurrence(event=Started(TrainingIteration(step=1)), count=1), context, state
    )
    with_sync.handle_occurrence(
        Occurrence(event=Ended(TrainingIteration(step=1)), count=1), context, state
    )
    assert calls == ["sync", "sync"]
