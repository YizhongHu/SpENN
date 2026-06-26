"""Unit tests for the canonical VMC objective, term metrics, and logabs summary."""

from __future__ import annotations

import importlib.util
import math

import pytest
import torch

from spenn.training.vmc import (
    VMCObjectiveResult,
    compute_vmc_objective,
    hamiltonian_term_metric_prefix,
    summarize_local_energy_terms,
    summarize_logabs,
)


def _logabs(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)


# --- compute_vmc_objective ---


def test_compute_vmc_objective_returns_differentiable_loss() -> None:
    logabs = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    local_energy = torch.tensor([1.0, 2.0, 3.0])

    result = compute_vmc_objective(logabs, local_energy)

    assert isinstance(result, VMCObjectiveResult)
    assert result.loss.requires_grad
    result.loss.backward()
    assert logabs.grad is not None
    assert torch.isfinite(logabs.grad).all()


def test_compute_vmc_objective_detaches_local_energy() -> None:
    logabs = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    local_energy = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    result = compute_vmc_objective(logabs, local_energy)
    result.loss.backward()

    assert local_energy.grad is None


def test_compute_vmc_objective_constant_energy_gives_zero_gradient() -> None:
    logabs = _logabs([0.1, -0.2, 0.3, -0.4])
    local_energy = torch.full((4,), 1.25, dtype=torch.float64)

    result = compute_vmc_objective(logabs, local_energy)
    result.loss.backward()

    # Centering a constant local energy zeroes the score-function weights.
    assert torch.allclose(logabs.grad, torch.zeros_like(logabs.grad))


def test_compute_vmc_objective_masks_nonfinite_local_energy() -> None:
    logabs = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    local_energy = torch.tensor([1.0, float("nan"), 3.0])

    result = compute_vmc_objective(logabs, local_energy)

    assert result.metrics["local_energy_n_finite"] == 2
    assert result.metrics["local_energy_n_total"] == 3
    assert result.metrics["local_energy_nonfinite_count"] == 1
    assert result.metrics["local_energy_finite_fraction"] == 2 / 3
    assert torch.isfinite(result.loss)


def test_compute_vmc_objective_raises_when_no_finite_local_energy() -> None:
    logabs = torch.tensor([0.1, 0.2], requires_grad=True)
    local_energy = torch.tensor([float("nan"), float("inf")])

    with pytest.raises(ValueError, match="no finite local-energy samples"):
        compute_vmc_objective(logabs, local_energy)


def test_compute_vmc_objective_raises_on_shape_mismatch() -> None:
    logabs = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    local_energy = torch.tensor([1.0, 2.0])

    with pytest.raises(ValueError, match="same shape"):
        compute_vmc_objective(logabs, local_energy)


def test_compute_vmc_objective_reports_json_safe_metrics() -> None:
    logabs = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    local_energy = torch.tensor([1.0, 2.0, 3.0])

    metrics = compute_vmc_objective(logabs, local_energy).metrics

    assert all(isinstance(value, (float, int)) for value in metrics.values())
    assert all(not isinstance(value, torch.Tensor) for value in metrics.values())


def test_compute_vmc_objective_uses_energy_metric_name() -> None:
    logabs = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    local_energy = torch.tensor([1.0, 2.0, 3.0])

    result = compute_vmc_objective(logabs, local_energy)

    assert "energy" in result.metrics
    assert "energy_mean" not in result.metrics
    assert result.metrics["energy"] == pytest.approx(2.0)
    assert result.metrics["energy_variance"] == pytest.approx(2.0 / 3.0)


# --- summarize_local_energy_terms ---


def test_summarize_local_energy_terms_names_terms_by_resolved_name() -> None:
    terms = {"kinetic": torch.tensor([1.0, 2.0]), "harmonic_trap": torch.tensor([3.0, 5.0])}

    metrics = summarize_local_energy_terms(terms)

    assert metrics["energy_term_kinetic"] == pytest.approx(1.5)
    assert metrics["energy_term_harmonic_trap"] == pytest.approx(4.0)


def test_summarize_local_energy_terms_distinguishes_terms_by_distinct_names() -> None:
    terms = {"harmonic_trap_0": torch.tensor([1.0, 2.0]), "harmonic_trap_1": torch.tensor([3.0, 5.0])}

    metrics = summarize_local_energy_terms(terms)

    assert metrics["energy_term_harmonic_trap_0"] == pytest.approx(1.5)
    assert metrics["energy_term_harmonic_trap_1"] == pytest.approx(4.0)


def test_summarize_local_energy_terms_reports_json_safe_health_metrics() -> None:
    metrics = summarize_local_energy_terms({"kinetic": torch.tensor([1.0, float("nan"), 3.0])})

    prefix = "energy_term_kinetic"
    assert metrics[f"{prefix}_n_finite"] == 2
    assert metrics[f"{prefix}_n_total"] == 3
    assert metrics[f"{prefix}_nonfinite_count"] == 1
    assert metrics[f"{prefix}_finite_fraction"] == pytest.approx(2 / 3)
    assert all(isinstance(value, (float, int)) for value in metrics.values())


def test_summarize_local_energy_terms_raises_when_no_finite_samples() -> None:
    with pytest.raises(ValueError, match="no finite samples"):
        summarize_local_energy_terms({"kinetic": torch.tensor([float("nan"), float("inf")])})


def test_hamiltonian_term_metric_prefix_uses_resolved_name() -> None:
    assert hamiltonian_term_metric_prefix("electron_electron") == "energy_term_electron_electron"


# --- old public surface is gone ---


def test_spenn_losses_vmc_loss_is_not_public_api() -> None:
    assert importlib.util.find_spec("spenn.losses") is None


# --- summarize_logabs ---


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
