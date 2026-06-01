"""Cusp-factor tests for trainable wavefunction ansatz modules."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data import FeatureDict
from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.nn.cusp import Cusp, ElectronElectronCusp, NuclearCusp, NuclearFeatureCusp
from spenn.nn.wavefunction import SpENNWavefunction
from spenn.physics.systems import ElectronicSystem


class EmptyEncoder(nn.Module):
    def forward(self, batch: ElectronBatch) -> FeatureDict:
        return FeatureDict()


class ConstantReadout(nn.Module):
    def forward(self, features: FeatureDict, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = torch.zeros(batch.batch_size, device=batch.device, dtype=batch.dtype)
        sign = torch.tensor([-1.0, 1.0], device=batch.device, dtype=batch.dtype)[: batch.batch_size]
        return WavefunctionOutput(logabs=logabs, sign=sign)


class BadShapeCusp(Cusp):
    def forward(self, batch: ElectronBatch) -> torch.Tensor:
        return torch.zeros(batch.batch_size, 1, device=batch.device, dtype=batch.dtype)


def test_spinless_electron_electron_cusp_matches_rational_option_a_formula() -> None:
    positions = torch.tensor([[[0.0], [2.0]], [[1.0], [4.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    cusp = ElectronElectronCusp(coefficient=0.25, range_parameter=0.5, eps=0.0)

    values = cusp(batch)

    distances = torch.tensor([2.0, 3.0], dtype=torch.float64)
    expected = 0.25 * distances / (1.0 + 0.5 * distances)
    assert values.shape == (2,)
    assert values.dtype == torch.float64
    assert values.device == positions.device
    assert torch.allclose(values, expected)


def test_electron_electron_cusp_is_permutation_invariant_and_has_short_range_slope() -> None:
    positions = torch.tensor([[[0.0], [1.0], [3.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    cusp = ElectronElectronCusp(coefficient=0.25, range_parameter=0.75, eps=0.0)

    permuted = ElectronBatch(positions=positions[:, [2, 0, 1]])

    assert torch.allclose(cusp(batch), cusp(permuted))

    tiny_r = torch.tensor(1.0e-7, dtype=torch.float64)
    near_coalescence = ElectronBatch(positions=torch.stack([torch.zeros_like(tiny_r), tiny_r]).view(1, 2, 1))
    slope = cusp(near_coalescence) / tiny_r
    assert torch.allclose(slope, torch.tensor([0.25], dtype=torch.float64), atol=1.0e-6, rtol=0.0)


def test_electron_electron_cusp_uses_spin_resolved_slopes() -> None:
    tiny_r = torch.tensor(1.0e-7, dtype=torch.float64)
    positions = torch.stack([torch.zeros_like(tiny_r), tiny_r]).view(1, 2, 1)
    same_spin = ElectronBatch(positions=positions, spins=torch.tensor([[1.0, 1.0]], dtype=torch.float64))
    opposite_spin = ElectronBatch(positions=positions, spins=torch.tensor([[1.0, -1.0]], dtype=torch.float64))
    cusp = ElectronElectronCusp(range_parameter=0.5, eps=0.0)

    same_slope = cusp(same_spin) / tiny_r
    opposite_slope = cusp(opposite_spin) / tiny_r

    assert torch.allclose(same_slope, torch.tensor([0.25], dtype=torch.float64), atol=1.0e-6, rtol=0.0)
    assert torch.allclose(opposite_slope, torch.tensor([0.5], dtype=torch.float64), atol=1.0e-6, rtol=0.0)


def test_electron_electron_trainable_ranges_are_positive_and_differentiable() -> None:
    positions = torch.tensor([[[0.0], [1.0], [2.0]]], dtype=torch.float64)
    spins = torch.tensor([[1.0, 1.0, -1.0]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions, spins=spins)
    cusp = ElectronElectronCusp(range_parameter=0.5, trainable_range=True, eps=1.0e-12)

    output = cusp(batch).sum()
    output.backward()

    assert torch.all(cusp.same_range_parameter > 0)
    assert torch.all(cusp.opposite_range_parameter > 0)
    assert cusp.raw_same_range.grad is not None
    assert cusp.raw_opposite_range.grad is not None


def test_nuclear_cusp_matches_rational_option_a_formula_and_slope() -> None:
    tiny_r = torch.tensor(1.0e-7, dtype=torch.float64)
    positions = torch.stack([tiny_r]).view(1, 1, 1)
    batch = ElectronBatch(positions=positions)
    cusp = NuclearCusp(
        nuclear_positions=torch.tensor([[0.0]], dtype=torch.float64),
        nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
        range_parameter=0.5,
        eps=0.0,
    )

    values = cusp(batch)
    slope = values / tiny_r

    assert values.shape == (1,)
    assert torch.allclose(values, -2.0 * tiny_r / (1.0 + 0.5 * tiny_r))
    assert torch.allclose(slope, torch.tensor([-2.0], dtype=torch.float64), atol=1.0e-6, rtol=0.0)


def test_nuclear_cusp_uses_batch_nuclear_data_when_constructor_data_is_absent() -> None:
    batch = ElectronBatch(
        positions=torch.tensor([[[1.0]], [[2.0]]], dtype=torch.float64),
        nuclear_positions=torch.tensor([[0.0]], dtype=torch.float64),
        nuclear_charges=torch.tensor([1.5], dtype=torch.float64),
    )
    cusp = NuclearCusp(range_parameter=0.25, eps=0.0)

    values = cusp(batch)
    distances = torch.tensor([1.0, 2.0], dtype=torch.float64)
    expected = -1.5 * distances / (1.0 + 0.25 * distances)

    assert torch.allclose(values, expected)


def test_nuclear_cusp_uses_system_nuclear_data_when_batch_data_is_absent() -> None:
    system = ElectronicSystem(
        n_electrons=1,
        spatial_dim=1,
        nuclear_positions=torch.tensor([[0.0]], dtype=torch.float64),
        nuclear_charges=torch.tensor([1.5], dtype=torch.float64),
    )
    batch = ElectronBatch(positions=torch.tensor([[[1.0]], [[2.0]]], dtype=torch.float64), system=system)
    cusp = NuclearCusp(range_parameter=0.25, eps=0.0)

    values = cusp(batch)
    distances = torch.tensor([1.0, 2.0], dtype=torch.float64)
    expected = -1.5 * distances / (1.0 + 0.25 * distances)

    assert torch.allclose(values, expected)


def test_nuclear_cusp_trainable_range_is_positive_and_differentiable() -> None:
    batch = ElectronBatch(
        positions=torch.tensor([[[1.0]], [[2.0]]], dtype=torch.float64),
        nuclear_positions=torch.tensor([[0.0]], dtype=torch.float64),
        nuclear_charges=torch.tensor([1.0], dtype=torch.float64),
    )
    cusp = NuclearCusp(range_parameter=0.5, trainable_range=True, eps=1.0e-12)

    output = cusp(batch).sum()
    output.backward()

    assert torch.all(cusp.range_parameter > 0)
    assert cusp.raw_range.grad is not None


def test_nuclear_feature_cusp_is_scaffold_only() -> None:
    batch = ElectronBatch(positions=torch.ones(2, 1, 3, dtype=torch.float64))

    try:
        NuclearFeatureCusp()(batch)
    except NotImplementedError as exc:
        assert "nuclear feature" in str(exc).lower()
    else:
        raise AssertionError("Expected NuclearFeatureCusp to raise NotImplementedError")


def test_disabled_cusp_returns_zero_batch_vector() -> None:
    batch = ElectronBatch(positions=torch.ones(4, 2, 3, dtype=torch.float64))

    values = ElectronElectronCusp(enabled=False)(batch)

    assert torch.equal(values, torch.zeros(4, dtype=torch.float64))


def test_wavefunction_cusp_adds_only_to_logabs_and_preserves_sign() -> None:
    positions = torch.tensor([[[0.0], [2.0]], [[1.0], [4.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    cusp = ElectronElectronCusp(coefficient=0.25, range_parameter=0.5, eps=0.0)
    model = SpENNWavefunction(encoder=EmptyEncoder(), spechtmp=nn.Identity(), readout=ConstantReadout(), cusp=cusp)

    output = model(batch)

    assert torch.allclose(output.logabs, cusp(batch))
    assert torch.equal(output.sign, torch.tensor([-1.0, 1.0], dtype=torch.float64))


def test_wavefunction_cusp_shape_must_match_readout_logabs() -> None:
    batch = ElectronBatch(positions=torch.ones(2, 2, 1, dtype=torch.float64))
    model = SpENNWavefunction(encoder=EmptyEncoder(), spechtmp=nn.Identity(), readout=ConstantReadout(), cusp=BadShapeCusp())

    try:
        model(batch)
    except ValueError as exc:
        assert "cusp output" in str(exc).lower()
    else:
        raise AssertionError("Expected mismatched cusp shape to raise ValueError")
