"""Unit tests for the VMC surrogate loss and log-amplitude summary."""

from __future__ import annotations

import math

import pytest
import torch

from spenn.training.vmc import summarize_logabs, vmc_surrogate_loss


def _logabs(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)


def test_finite_energy_gives_finite_scalar_loss() -> None:
    logabs = _logabs([0.1, -0.2, 0.3, -0.4])
    energy = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float64)

    loss = vmc_surrogate_loss(logabs, energy)

    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_constant_energy_gives_zero_gradient_signal() -> None:
    logabs = _logabs([0.1, -0.2, 0.3, -0.4])
    energy = torch.full((4,), 1.25, dtype=torch.float64)

    loss = vmc_surrogate_loss(logabs, energy)
    loss.backward()

    # Centering a constant energy zeroes the score-function weights.
    assert torch.allclose(logabs.grad, torch.zeros_like(logabs.grad))


def test_nonconstant_energy_gives_nonzero_gradient_on_logabs() -> None:
    logabs = _logabs([0.1, -0.2, 0.3, -0.4])
    energy = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float64)

    loss = vmc_surrogate_loss(logabs, energy)
    loss.backward()

    assert logabs.grad is not None
    assert torch.any(logabs.grad != 0)


def test_energy_is_detached_no_grad_required() -> None:
    logabs = _logabs([0.1, -0.2, 0.3])
    energy = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64, requires_grad=True)

    loss = vmc_surrogate_loss(logabs, energy)
    loss.backward()

    assert energy.grad is None


def test_masks_nonfinite_energy_samples() -> None:
    logabs = _logabs([0.1, -0.2, 0.3, -0.4])
    energy = torch.tensor([1.0, float("nan"), 0.5, float("inf")], dtype=torch.float64)

    loss = vmc_surrogate_loss(logabs, energy)

    assert torch.isfinite(loss)


def test_all_nonfinite_energy_raises_runtime_error() -> None:
    logabs = _logabs([0.1, -0.2])
    energy = torch.tensor([float("nan"), float("inf")], dtype=torch.float64)

    with pytest.raises(RuntimeError):
        vmc_surrogate_loss(logabs, energy)


def test_summarize_logabs_returns_finite_python_floats() -> None:
    logabs = torch.tensor([0.0, 1.0, -1.0, 2.0], dtype=torch.float64)

    summary = summarize_logabs(logabs)

    assert set(summary) == {"logabs_mean", "logabs_min", "logabs_max", "nonfinite_logabs_fraction"}
    assert all(isinstance(value, float) for value in summary.values())
    assert summary["logabs_min"] == -1.0
    assert summary["logabs_max"] == 2.0
    assert summary["nonfinite_logabs_fraction"] == 0.0


def test_summarize_logabs_reports_nonfinite_fraction() -> None:
    logabs = torch.tensor([0.0, float("inf"), float("nan"), 2.0], dtype=torch.float64)

    summary = summarize_logabs(logabs)

    assert summary["nonfinite_logabs_fraction"] == 0.5
    assert math.isfinite(summary["logabs_mean"])
