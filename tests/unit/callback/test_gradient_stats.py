"""Tests for the GradientStats runtime-check callback."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tpen.callback import GradientStats
from tests.unit.callback.support import (
    RecordingContext,
    deliver_completed_iteration,
    training_state,
)

STEP = 4


class TwoParamModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros(2, dtype=torch.float64))


def _handle(callback: GradientStats, model: nn.Module) -> RecordingContext:
    context = RecordingContext()
    # ``state.step`` is left at its constructor default on purpose: the logged
    # coordinate must come from the typed event, so a state carrying a
    # different value would expose a callback that read the wrong one.
    deliver_completed_iteration(
        callback, context, training_state(model=model), step=STEP
    )
    return context


def test_logs_global_grad_norm() -> None:
    model = TwoParamModule()
    model.a.grad = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float64)
    model.b.grad = torch.zeros(2, dtype=torch.float64)

    metrics = _handle(GradientStats(), model).latest("checks/gradient")

    assert metrics["global_grad_norm"] == pytest.approx(5.0)
    assert metrics["max_abs_grad"] == pytest.approx(4.0)
    assert metrics["n_grad_tensors"] == 2
    assert metrics["n_grad_elements"] == 5
    assert metrics["nonfinite_grad_fraction"] == 0.0
    assert metrics["passed"] is True


def test_handles_parameters_with_grad_none() -> None:
    model = TwoParamModule()
    model.a.grad = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    # model.b.grad stays None

    metrics = _handle(GradientStats(), model).latest("checks/gradient")

    assert metrics["n_grad_tensors"] == 1
    assert metrics["n_grad_elements"] == 3
    assert metrics["global_grad_norm"] == pytest.approx(1.0)


def test_no_gradients_passes_with_zero_norm() -> None:
    metrics = _handle(GradientStats(), TwoParamModule()).latest("checks/gradient")

    assert metrics["n_grad_tensors"] == 0
    assert metrics["global_grad_norm"] == 0.0
    assert metrics["passed"] is True


def test_detects_nonfinite_gradients_without_raising() -> None:
    model = TwoParamModule()
    model.a.grad = torch.tensor([1.0, float("inf"), 2.0], dtype=torch.float64)

    metrics = _handle(GradientStats(fail_fast=False), model).latest("checks/gradient")

    assert metrics["nonfinite_grad_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["passed"] is False
    assert metrics["global_grad_norm"] == pytest.approx(5.0**0.5)  # finite-only norm


def test_fail_fast_raises_on_nonfinite_gradients() -> None:
    model = TwoParamModule()
    model.a.grad = torch.tensor([float("nan"), 0.0, 0.0], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="nonfinite_grad_fraction"):
        _handle(GradientStats(fail_fast=True, check_finite=True), model)


def test_fail_fast_raises_when_grad_norm_exceeds_max() -> None:
    model = TwoParamModule()
    model.a.grad = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="global_grad_norm"):
        _handle(GradientStats(fail_fast=True, max_global_grad_norm=1.0), model)


def test_logged_step_comes_from_the_event_not_the_state() -> None:
    # `TrainerState` value fields are assigned at the END of the trainer's loop
    # body, so at any earlier boundary they hold the previous iteration's values
    # and, on iteration 0, the constructor default -1. A callback that took its
    # coordinate from the state would publish that stale number. Measured on
    # Cannon: a probe on both iteration boundaries read
    # ``[-1, 0, 0, 1, 1, 2, ...]`` while the metric axis ran a clean 0..9.
    context = RecordingContext()
    state = training_state(model=TwoParamModule(), step=-1)

    deliver_completed_iteration(context=context, callback=GradientStats(), state=state, step=7)

    assert [record["step"] for record in context.by_namespace("checks/gradient")] == [7]


def test_cadence_gates_on_the_durable_step_not_the_occurrence_count() -> None:
    # A typed `Cadence` would gate on ``Occurrence.count``, which is run-local
    # and restarts at 1 after a checkpoint restore. Holding the count fixed
    # while the durable step advances is exactly the situation a resumed run
    # creates, and the schedule must follow the step.
    callback = GradientStats(every_n_steps=5)
    context = RecordingContext()
    state = training_state(model=TwoParamModule())

    for step in range(12):
        deliver_completed_iteration(callback, context, state, step=step, count=1)

    assert [record["step"] for record in context.by_namespace("checks/gradient")] == [0, 5, 10]
