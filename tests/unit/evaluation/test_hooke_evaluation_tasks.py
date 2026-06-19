"""Tests for Hooke-specific evaluation generators and summaries."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import spenn.diagnostics as diagnostics
from spenn.data.batch import ElectronBatch
from spenn.evaluation.bundle import (
    DerivativeValues,
    EvaluationBundle,
    GeneratedConfigurations,
    LocalEnergyValues,
    WavefunctionValues,
)
from spenn.evaluation.generators import CuspGridGenerator, TailGridGenerator
from spenn.evaluation.protocols import EvaluationContext
from spenn.evaluation.summaries import (
    CoalescenceDivergenceSummary,
    LocalEnergyStabilitySummary,
    OppositeSpinCuspSummary,
    PathologyCountSummary,
)


def _context(tmp_path: Path) -> EvaluationContext:
    return EvaluationContext(
        namespace="validation/hooke",
        artifact_level="metrics_only",
        task_failure_policy="continue",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=0,
        run_dir=tmp_path,
        task_output_dir=tmp_path,
        metadata={},
    )


def _batch(n_samples: int) -> ElectronBatch:
    positions = torch.zeros(n_samples, 2, 3, dtype=torch.float64)
    spins = torch.tensor([[1.0, -1.0]], dtype=torch.float64).repeat(n_samples, 1)
    return ElectronBatch(positions=positions, spins=spins)


def test_cusp_grid_generator_pairs_antipodal_directions(tmp_path: Path) -> None:
    generated = CuspGridGenerator(
        n_points=3,
        r12_min=0.1,
        r12_max=1.0,
        n_directions=2,
        center_of_mass_radii=[0.0],
        paired_directions=True,
    ).generate(model=None, context=_context(tmp_path))

    assert generated.batch.positions.shape == (12, 2, 3)
    assert set(generated.metadata["direction_sign"].tolist()) == {-1, 1}
    pair_id = generated.metadata["antipodal_pair_id"]
    direction_sign = generated.metadata["direction_sign"]
    for value in torch.unique(pair_id):
        mask = pair_id == value
        assert int((direction_sign[mask] > 0).sum().item()) == 3
        assert int((direction_sign[mask] < 0).sum().item()) == 3


def test_opposite_spin_cusp_summary_uses_even_odd_pairs(tmp_path: Path) -> None:
    radial = torch.tensor([0.51, 0.49, 0.52, 0.48], dtype=torch.float64)
    bundle = EvaluationBundle(
        generated=GeneratedConfigurations(
            batch=_batch(4),
            metadata={"spin_pair": "opposite"},
        ),
        derivatives={
            "r12": DerivativeValues(
                radial_dlogabs=radial,
                r12=torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float64),
                direction_id=torch.tensor([0, 0, 1, 1]),
                antipodal_pair_id=torch.tensor([0, 0, 1, 1]),
                direction_sign=torch.tensor([1, -1, 1, -1]),
            )
        },
    )

    metrics = OppositeSpinCuspSummary().summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="validation/cusp",
    ).metrics

    assert metrics["cusp_even_slope_mean"] == pytest.approx(0.5)
    assert metrics["cusp_even_slope_abs_error"] == pytest.approx(0.0)
    assert metrics["cusp_odd_slant_mean_abs"] == pytest.approx(0.015)


def test_coalescence_divergence_summary_fits_zero_cminus_one(tmp_path: Path) -> None:
    r12 = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    local_energy = 2.0 + 0.25 * r12
    bundle = EvaluationBundle(
        generated=GeneratedConfigurations(
            batch=_batch(4),
            metadata={
                "r12": r12,
                "direction_id": torch.zeros(4, dtype=torch.long),
                "center_of_mass_id": torch.zeros(4, dtype=torch.long),
            },
        ),
        local_energy=LocalEnergyValues(
            local_energy=local_energy,
            finite_mask=torch.isfinite(local_energy),
            term_energies=None,
        ),
    )

    metrics = CoalescenceDivergenceSummary().summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="validation/coalescence",
    ).metrics

    assert metrics["c_minus_1_abs_max"] == pytest.approx(0.0, abs=1.0e-10)
    assert metrics["coalescence_fit_failure_count"] == 0


def test_tail_grid_and_pathology_summaries_report_finite_metrics(tmp_path: Path) -> None:
    generated = TailGridGenerator(
        radius_min=1.0,
        radius_max=2.0,
        n_points=3,
        pair_distance=0.5,
        n_directions=1,
    ).generate(model=None, context=_context(tmp_path))
    local_energy = torch.tensor([2.0, 2.1, float("nan")], dtype=torch.float64)
    bundle = EvaluationBundle(
        generated=generated,
        local_energy=LocalEnergyValues(
            local_energy=local_energy,
            finite_mask=torch.isfinite(local_energy),
            term_energies=None,
        ),
        wavefunction=WavefunctionValues(
            logabs=torch.tensor([0.0, -1.0, -2.0], dtype=torch.float64),
            sign=torch.ones(3, dtype=torch.float64),
        ),
    )

    stability_metrics = LocalEnergyStabilitySummary(abs_threshold=10.0).summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="validation/tail",
    ).metrics
    pathology_metrics = PathologyCountSummary().summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="validation/pathology",
    ).metrics

    assert generated.batch.positions.shape == (3, 2, 3)
    assert stability_metrics["stability_outlier_count"] == 0
    assert stability_metrics["stability_n_finite"] == 2
    assert pathology_metrics["nonfinite_local_energy_count"] == 1


def test_local_energy_stability_summary_requires_explicit_threshold() -> None:
    with pytest.raises(TypeError):
        LocalEnergyStabilitySummary()  # type: ignore[call-arg]


def test_tail_grid_log_spacing_requires_positive_radius_min() -> None:
    with pytest.raises(ValueError, match="radius_min"):
        TailGridGenerator(
            radius_min=0.0,
            radius_max=2.0,
            n_points=3,
            pair_distance=0.5,
            n_directions=1,
            spacing="log",
        )


def test_old_hooke_probe_names_are_not_public() -> None:
    assert not hasattr(diagnostics, "Hooke" + "Pair" + "DistanceProbe")
    assert not hasattr(diagnostics, "Hooke" + "Pair" + "CenterOfMassProbe")
