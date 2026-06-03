"""Tests for reusable multibody wavefunction diagnostics."""

from __future__ import annotations

import math

import torch
from torch import nn
from omegaconf import OmegaConf

from spenn.data.batch import ElectronBatch, Walkers, WavefunctionOutput
from spenn.diagnostics.base import DiagnosticContext
from spenn.diagnostics.wavefunction import (
    PairDistanceHistogramDiagnostic,
    ParticleAntisymmetryDiagnostic,
    RadialDensityDiagnostic,
    SpinResolvedCuspSlopeDiagnostic,
    all_pair_distances,
)
from spenn.physics.systems import ElectronicSystem


def test_all_pair_distances_preserves_two_electron_behavior_and_shape() -> None:
    positions = torch.tensor(
        [
            [[0.0, 0.0], [3.0, 4.0]],
            [[1.0, 1.0], [1.0, 3.0]],
        ],
        dtype=torch.float64,
    )

    distances = all_pair_distances(positions)

    assert distances.shape == (2,)
    assert distances.dtype == positions.dtype
    assert distances.device == positions.device
    assert torch.allclose(distances, torch.tensor([5.0, 2.0], dtype=torch.float64))


def test_all_pair_distances_flattens_upper_triangle_for_three_electrons() -> None:
    positions = torch.tensor(
        [[[0.0, 0.0], [3.0, 4.0], [0.0, 12.0]]],
        dtype=torch.float64,
    )

    distances = all_pair_distances(positions)

    expected = torch.tensor([5.0, 12.0, torch.sqrt(torch.tensor(73.0, dtype=torch.float64))])
    assert distances.shape == (3,)
    assert torch.allclose(distances, expected)


def test_all_pair_distances_preserves_float32_dtype() -> None:
    positions = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]],
        dtype=torch.float32,
    )

    distances = all_pair_distances(positions)

    assert distances.dtype == torch.float32
    assert distances.device == positions.device
    assert distances.shape == (3,)


def test_all_pair_distances_rejects_single_electron_inputs() -> None:
    positions = torch.zeros(2, 1, 3)

    try:
        all_pair_distances(positions)
    except ValueError as exc:
        assert "at least two electrons" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("single-electron pair distances should fail")


def test_radial_density_diagnostic_emits_histogram_and_mean_radius() -> None:
    context = _context(
        model=ConstantModel(),
        positions=torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            ],
            dtype=torch.float64,
        ),
        spins=torch.tensor([[1.0, 1.0, -1.0]], dtype=torch.float64),
    )

    result = RadialDensityDiagnostic(bins=3)(context)

    assert "radial_density/mean_radius" in result.metrics
    assert result.metrics["radial_density/mean_radius"] == 2.0
    rows = result.tables["radial_density"]
    assert len(rows) == 3
    assert sum(row["count"] for row in rows) == 3.0
    assert all("probability_density" in row for row in rows)


def test_pair_distance_histogram_diagnostic_uses_context_pair_distances() -> None:
    context = _context(
        model=ConstantModel(),
        positions=torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            ],
            dtype=torch.float64,
        ),
        spins=torch.tensor([[1.0, 1.0, -1.0]], dtype=torch.float64),
    )

    result = PairDistanceHistogramDiagnostic(bins=2)(context)

    rows = result.tables["pair_distance_histogram"]
    assert len(rows) == 2
    assert sum(row["count"] for row in rows) == 3.0


def test_spin_resolved_cusp_diagnostic_groups_same_and_opposite_spin_pairs() -> None:
    positions = torch.zeros(2, 2, 3, dtype=torch.float64)
    spins = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    context = _context(model=SpinResolvedCuspToyModel(), positions=positions, spins=spins, n_up=1, n_down=1)

    result = SpinResolvedCuspSlopeDiagnostic(n_points=8, n_configurations=2)(context)

    assert abs(result.metrics["cusp/same_mean_error"]) < 1.0e-10
    assert abs(result.metrics["cusp/opposite_mean_error"]) < 1.0e-10
    relations = {row["spin_relation"] for row in result.tables["cusp_slope_by_spin"]}
    assert relations == {"same", "opposite"}
    assert result.metrics["cusp/same_count"] == 1.0
    assert result.metrics["cusp/opposite_count"] == 1.0


def test_spin_resolved_cusp_diagnostic_keeps_stable_keys_for_absent_relations() -> None:
    positions = torch.zeros(1, 3, 3, dtype=torch.float64)
    spins = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    context = _context(model=SpinResolvedCuspToyModel(), positions=positions, spins=spins, n_up=3, n_down=0)

    result = SpinResolvedCuspSlopeDiagnostic(n_points=8, n_configurations=1)(context)

    assert result.metrics["cusp/same_count"] == 3.0
    assert result.metrics["cusp/opposite_count"] == 0.0
    assert math.isfinite(result.metrics["cusp/same_mean_error"])
    assert math.isnan(result.metrics["cusp/opposite_mean_error"])
    assert math.isnan(result.metrics["cusp/opposite_max_abs_error"])


def test_particle_antisymmetry_diagnostic_checks_token_transpositions() -> None:
    positions = torch.tensor(
        [
            [[-1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[-2.0, 0.0, 0.0], [0.25, 0.0, 0.0], [1.5, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    spins = torch.tensor([[1.0, 1.0, -1.0], [1.0, 1.0, -1.0]], dtype=torch.float64)
    context = _context(model=VandermondeModel(), positions=positions, spins=spins, n_up=2, n_down=1)

    result = ParticleAntisymmetryDiagnostic(n_samples=2, max_transpositions=3)(context)

    assert result.metrics["antisymmetry/antisymmetry_error_max"] < 1.0e-12
    assert result.metrics["antisymmetry/sign_flip_accuracy"] == 1.0
    assert len(result.tables["particle_antisymmetry"]) == 3


class ConstantModel(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        flat = batch.flatten_samples()
        logabs = torch.zeros(flat.batch_size, device=flat.device, dtype=flat.dtype)
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class SpinResolvedCuspToyModel(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        flat = batch.flatten_samples()
        distance = torch.linalg.norm(flat.positions[:, 0] - flat.positions[:, 1], dim=-1)
        same_spin = flat.spins[:, 0] == flat.spins[:, 1]
        same_logabs = torch.log(distance.clamp_min(torch.finfo(flat.dtype).tiny)) + 0.25 * distance
        opposite_logabs = 0.5 * distance
        logabs = torch.where(same_spin, same_logabs, opposite_logabs)
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class VandermondeModel(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        flat = batch.flatten_samples()
        x = flat.positions[..., 0]
        value = torch.ones(flat.batch_size, device=flat.device, dtype=flat.dtype)
        for i in range(flat.n_electrons):
            for j in range(i + 1, flat.n_electrons):
                value = value * (x[:, i] - x[:, j])
        sign = torch.sign(value)
        logabs = torch.log(value.abs())
        return WavefunctionOutput(logabs=logabs, sign=sign)


def _context(
    *,
    model: nn.Module,
    positions: torch.Tensor,
    spins: torch.Tensor,
    n_up: int | None = None,
    n_down: int | None = None,
) -> DiagnosticContext:
    system = ElectronicSystem(
        n_electrons=positions.shape[1],
        spatial_dim=positions.shape[2],
        n_up=n_up,
        n_down=n_down,
        dtype=positions.dtype,
    )
    walkers = Walkers(positions=positions, spins=spins, aux={"system": system})
    return DiagnosticContext(
        cfg=OmegaConf.create({}),
        model=model,
        hamiltonian=None,
        system=system,
        sampler=None,
        walkers=walkers,
        local_energy=torch.zeros(positions.shape[0], dtype=positions.dtype),
        pair_distance=all_pair_distances(positions),
        dtype=positions.dtype,
        device=positions.device,
    )
