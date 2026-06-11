"""Unit tests for evaluation energy diagnostics."""

from __future__ import annotations

import math

import pytest
import torch

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.diagnostics import EnergyEvaluation, EvaluationContext


def _context(
    local_energy: torch.Tensor,
    *,
    terms: dict[str, torch.Tensor] | None = None,
) -> EvaluationContext:
    batch = ElectronBatch(positions=torch.zeros(local_energy.numel(), 2, 1, dtype=torch.float64))
    output = WavefunctionOutput(
        logabs=torch.zeros(local_energy.numel(), dtype=torch.float64),
        sign=torch.ones(local_energy.numel(), dtype=torch.float64),
    )
    return EvaluationContext(
        model=object(),
        batch=batch,
        wavefunction_output=output,
        local_energy=local_energy,
        local_energy_terms=terms,
        sampler_stats={},
        hamiltonian_terms={},
    )


def test_energy_evaluation_masks_nonfinite_samples() -> None:
    context = _context(torch.tensor([1.0, float("nan"), 3.0, float("inf")], dtype=torch.float64))

    metrics = EnergyEvaluation().evaluate(context)

    assert metrics["energy"] == pytest.approx(2.0)
    assert metrics["energy_variance"] == pytest.approx(1.0)
    assert metrics["energy_std"] == pytest.approx(1.0)
    assert metrics["energy_stderr"] == pytest.approx(1.0 / math.sqrt(2.0))
    assert metrics["local_energy_n_finite"] == 2
    assert metrics["local_energy_n_total"] == 4
    assert metrics["local_energy_finite_fraction"] == pytest.approx(0.5)
    assert metrics["local_energy_nonfinite_count"] == 2


def test_energy_evaluation_raises_when_no_finite_samples() -> None:
    context = _context(torch.tensor([float("nan"), float("inf")], dtype=torch.float64))

    with pytest.raises(ValueError, match="no finite local-energy samples"):
        EnergyEvaluation().evaluate(context)


def test_energy_evaluation_reports_reference_error() -> None:
    context = _context(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    metrics = EnergyEvaluation(reference_energy=1.5).evaluate(context)

    assert metrics["energy"] == pytest.approx(2.0)
    assert metrics["energy_error"] == pytest.approx(0.5)
    assert metrics["energy_abs_error"] == pytest.approx(0.5)
    assert "energy_relative_error" not in metrics


def test_energy_evaluation_includes_configured_term_metrics() -> None:
    context = _context(
        torch.tensor([3.0, 5.0], dtype=torch.float64),
        terms={
            "kinetic": torch.tensor([1.0, 3.0], dtype=torch.float64),
            "harmonic_trap": torch.tensor([2.0, float("nan"), 4.0], dtype=torch.float64),
        },
    )

    metrics = EnergyEvaluation(include_terms=True).evaluate(context)

    assert metrics["energy_term_kinetic"] == pytest.approx(2.0)
    assert metrics["energy_term_kinetic_variance"] == pytest.approx(1.0)
    assert metrics["energy_term_harmonic_trap"] == pytest.approx(3.0)
    assert metrics["energy_term_harmonic_trap_n_finite"] == 2
    assert metrics["energy_term_harmonic_trap_n_total"] == 3


def test_energy_evaluation_include_terms_requires_returned_terms() -> None:
    context = _context(torch.tensor([1.0, 2.0], dtype=torch.float64), terms=None)

    with pytest.raises(ValueError, match="local_energy_terms"):
        EnergyEvaluation(include_terms=True).evaluate(context)
