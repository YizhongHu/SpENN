"""Unit tests for evaluation energy diagnostics."""

from __future__ import annotations

import math
import csv

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
    assert metrics["local_energy_min"] == pytest.approx(1.0)
    assert metrics["local_energy_max"] == pytest.approx(3.0)
    assert metrics["local_energy_q50"] == pytest.approx(2.0)


def test_energy_evaluation_raises_when_no_finite_samples() -> None:
    context = _context(torch.tensor([float("nan"), float("inf")], dtype=torch.float64))

    with pytest.raises(ValueError, match="no finite local-energy samples"):
        EnergyEvaluation().evaluate(context)


def test_energy_evaluation_reports_reference_error() -> None:
    context = _context(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    metrics = EnergyEvaluation(reference_energy=1.5).evaluate(context)

    assert metrics["energy"] == pytest.approx(2.0)
    assert metrics["reference_energy"] == pytest.approx(1.5)
    assert metrics["energy_error"] == pytest.approx(0.5)
    assert metrics["energy_abs_error"] == pytest.approx(0.5)
    assert "energy_relative_error" not in metrics


def test_energy_evaluation_reports_local_energy_error_summaries() -> None:
    context = _context(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    metrics = EnergyEvaluation(
        reference_energy=2.0,
        quantiles=[0.5],
        include_local_energy_error_quantiles=True,
    ).evaluate(context)

    assert metrics["local_energy_error_q50"] == pytest.approx(0.0)
    assert metrics["local_energy_error_mean"] == pytest.approx(0.0)
    assert metrics["local_energy_abs_error_mean"] == pytest.approx(2.0 / 3.0)


def test_energy_evaluation_does_not_write_sample_artifact_by_default(tmp_path) -> None:
    context = _context(torch.tensor([1.0, 2.0], dtype=torch.float64))
    context = EvaluationContext(**{**context.__dict__, "run_dir": tmp_path})

    EnergyEvaluation().evaluate(context)

    assert not (tmp_path / "diagnostics").exists()


def test_energy_evaluation_writes_optional_sampled_eval_table(tmp_path) -> None:
    local_energy = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    context = _context(
        local_energy,
        terms={
            "kinetic": local_energy + 10.0,
            "harmonic_trap": local_energy + 20.0,
            "electron_electron": local_energy + 30.0,
        },
    )
    context = EvaluationContext(**{**context.__dict__, "run_dir": tmp_path})

    EnergyEvaluation(
        reference_energy=2.0,
        sampled_eval_table={"enabled": True, "max_samples": 2, "selection": "stride"},
    ).evaluate(context)

    table = tmp_path / "diagnostics" / "energy" / "sampled_eval_table.csv"
    assert table.exists()
    with table.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_index"] for row in rows] == ["0", "2"]
    assert rows[0]["kinetic_energy"] == "11.0"
    index = tmp_path / "diagnostics" / "index.json"
    assert index.exists()


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
