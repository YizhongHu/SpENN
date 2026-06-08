"""Tests for the GradientStats runtime-check callback."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spenn.callback import GradientStats
from tests.unit.callback.support import FakeState, RecordingContext, step_event


class TwoParamModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros(2, dtype=torch.float64))


def _handle(callback: GradientStats, model: nn.Module) -> RecordingContext:
    context = RecordingContext()
    callback.handle(step_event(context, FakeState(model=model)))
    return context


def test_logs_global_grad_norm() -> None:
    model = TwoParamModule()
    model.a.grad = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float64)
    model.b.grad = torch.zeros(2, dtype=torch.float64)

    metrics = _handle(GradientStats(["step_end"]), model).latest("checks/gradient")

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

    metrics = _handle(GradientStats(["step_end"]), model).latest("checks/gradient")

    assert metrics["n_grad_tensors"] == 1
    assert metrics["n_grad_elements"] == 3
    assert metrics["global_grad_norm"] == pytest.approx(1.0)


def test_no_gradients_passes_with_zero_norm() -> None:
    metrics = _handle(GradientStats(["step_end"]), TwoParamModule()).latest("checks/gradient")

    assert metrics["n_grad_tensors"] == 0
    assert metrics["global_grad_norm"] == 0.0
    assert metrics["passed"] is True


def test_detects_nonfinite_gradients_without_raising() -> None:
    model = TwoParamModule()
    model.a.grad = torch.tensor([1.0, float("inf"), 2.0], dtype=torch.float64)

    metrics = _handle(GradientStats(["step_end"], fail_fast=False), model).latest("checks/gradient")

    assert metrics["nonfinite_grad_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["passed"] is False
    assert metrics["global_grad_norm"] == pytest.approx(5.0**0.5)  # finite-only norm


def test_fail_fast_raises_on_nonfinite_gradients() -> None:
    model = TwoParamModule()
    model.a.grad = torch.tensor([float("nan"), 0.0, 0.0], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="nonfinite_grad_fraction"):
        _handle(GradientStats(["step_end"], fail_fast=True, check_finite=True), model)


def test_fail_fast_raises_when_grad_norm_exceeds_max() -> None:
    model = TwoParamModule()
    model.a.grad = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="global_grad_norm"):
        _handle(GradientStats(["step_end"], fail_fast=True, max_global_grad_norm=1.0), model)
