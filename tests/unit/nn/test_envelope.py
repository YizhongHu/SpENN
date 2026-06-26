"""Envelope-factor tests for trainable wavefunction ansatz modules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.data.real import RealFeature
from spenn.nn import AdditiveEnvelope, ElectronElectronCusp, Envelope, HarmonicConfinement, SpENNWaveFunction
from tests.helpers.equivariance import assert_equivariant_all


class EmptyEncoder(nn.Module):
    def forward(self, batch: ElectronBatch) -> RealFeature:
        return RealFeature()


class ConstantReadout(nn.Module):
    def forward(self, features: RealFeature, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = torch.zeros(batch.batch_size, device=batch.device, dtype=batch.dtype)
        sign = torch.tensor([-1.0, 1.0], device=batch.device, dtype=batch.dtype)[: batch.batch_size]
        return WavefunctionOutput(logabs=logabs, sign=sign)


class AntisymmetricReadout(nn.Module):
    def forward(self, features: RealFeature, batch: ElectronBatch) -> WavefunctionOutput:
        sign = torch.sign(batch.positions[:, 0, 0] - batch.positions[:, 1, 0])
        return WavefunctionOutput(logabs=torch.zeros_like(sign), sign=sign)


class BadShapeEnvelope(Envelope):
    def envelope_value(self, batch: ElectronBatch) -> torch.Tensor:
        return torch.zeros(batch.batch_size, 1, device=batch.device, dtype=batch.dtype)


class FullOutputEnvelope(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = torch.zeros(batch.batch_size, device=batch.device, dtype=batch.dtype)
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def test_harmonic_confinement_matches_gaussian_tail_formula() -> None:
    positions = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    envelope = HarmonicConfinement(coefficient=0.25)

    values = envelope(batch)

    expected = -0.25 * positions.square().sum(dim=(1, 2))
    torch.testing.assert_close(values, expected)


def test_harmonic_confinement_is_permutation_invariant() -> None:
    positions = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    permuted = ElectronBatch(positions=positions[:, [2, 0, 1]])
    envelope = HarmonicConfinement(coefficient=0.25)

    torch.testing.assert_close(envelope(batch), envelope(permuted))


def test_harmonic_confinement_trainable_coefficient_is_nonnegative_and_differentiable() -> None:
    positions = torch.tensor([[[1.0], [2.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    envelope = HarmonicConfinement(coefficient=0.25, trainable=True)

    output = envelope(batch).sum()
    output.backward()

    assert torch.all(envelope.coefficient >= 0.0)
    assert envelope.raw_coefficient.grad is not None


def test_spinless_electron_electron_cusp_matches_rational_option_a_formula() -> None:
    positions = torch.tensor([[[0.0], [2.0]], [[1.0], [4.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    envelope = ElectronElectronCusp(spinless_coefficient=0.25, range_parameter=0.5, eps=0.0)

    values = envelope(batch)

    distances = torch.tensor([2.0, 3.0], dtype=torch.float64)
    expected = 0.25 * distances / (1.0 + 0.5 * distances)
    torch.testing.assert_close(values, expected)


def test_electron_electron_cusp_is_permutation_invariant_and_has_short_range_slope() -> None:
    positions = torch.tensor([[[0.0], [1.0], [3.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    envelope = ElectronElectronCusp(spinless_coefficient=0.25, range_parameter=0.75, eps=0.0)
    permuted = ElectronBatch(positions=positions[:, [2, 0, 1]])

    torch.testing.assert_close(envelope(batch), envelope(permuted))

    tiny_r = torch.tensor(1.0e-7, dtype=torch.float64)
    near_coalescence = ElectronBatch(positions=torch.stack([torch.zeros_like(tiny_r), tiny_r]).view(1, 2, 1))
    slope = envelope(near_coalescence) / tiny_r
    torch.testing.assert_close(slope, torch.tensor([0.25], dtype=torch.float64), atol=1.0e-6, rtol=0.0)


def test_electron_electron_cusp_uses_spin_resolved_slopes() -> None:
    tiny_r = torch.tensor(1.0e-7, dtype=torch.float64)
    positions = torch.stack([torch.zeros_like(tiny_r), tiny_r]).view(1, 2, 1)
    same_spin = ElectronBatch(positions=positions, spins=torch.tensor([[1.0, 1.0]], dtype=torch.float64))
    opposite_spin = ElectronBatch(positions=positions, spins=torch.tensor([[1.0, -1.0]], dtype=torch.float64))
    envelope = ElectronElectronCusp(range_parameter=0.5, eps=0.0)

    same_slope = envelope(same_spin) / tiny_r
    opposite_slope = envelope(opposite_spin) / tiny_r

    torch.testing.assert_close(same_slope, torch.tensor([0.25], dtype=torch.float64), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(opposite_slope, torch.tensor([0.5], dtype=torch.float64), atol=1.0e-6, rtol=0.0)


def test_electron_electron_trainable_ranges_are_positive_and_differentiable() -> None:
    positions = torch.tensor([[[0.0], [1.0], [2.0]]], dtype=torch.float64)
    spins = torch.tensor([[1.0, 1.0, -1.0]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions, spins=spins)
    envelope = ElectronElectronCusp(range_parameter=0.5, trainable_range=True, eps=1.0e-12)

    output = envelope(batch).sum()
    output.backward()

    assert torch.all(envelope.same_range_parameter > 0)
    assert torch.all(envelope.opposite_range_parameter > 0)
    assert envelope.raw_same_range.grad is not None
    assert envelope.raw_opposite_range.grad is not None


def test_disabled_envelope_returns_zero_batch_vector() -> None:
    batch = ElectronBatch(positions=torch.ones(4, 2, 3, dtype=torch.float64))

    values = HarmonicConfinement(enabled=False, coefficient=0.25)(batch)

    torch.testing.assert_close(values, torch.zeros(4, dtype=torch.float64))


def test_additive_envelope_sums_component_outputs() -> None:
    positions = torch.tensor([[[0.0], [2.0]], [[1.0], [4.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    harmonic = HarmonicConfinement(coefficient=0.25)
    cusp = ElectronElectronCusp(spinless_coefficient=0.25, range_parameter=0.5, eps=0.0)
    envelope = AdditiveEnvelope([harmonic, cusp])

    torch.testing.assert_close(envelope(batch), harmonic(batch) + cusp(batch))


def test_empty_additive_envelope_returns_zero_batch_vector() -> None:
    batch = ElectronBatch(positions=torch.ones(4, 2, 3, dtype=torch.float64))

    values = AdditiveEnvelope()(batch)

    torch.testing.assert_close(values, torch.zeros(4, dtype=torch.float64))


def test_wavefunction_requires_envelope() -> None:
    with pytest.raises(ValueError, match="envelope"):
        SpENNWaveFunction(
            embedding=EmptyEncoder(),
            layers=[nn.Identity()],
            readout=ConstantReadout(),
            envelope=None,  # type: ignore[arg-type]
        )


def test_wavefunction_envelope_adds_only_to_logabs_and_preserves_sign() -> None:
    positions = torch.tensor([[[0.0], [2.0]], [[1.0], [4.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    envelope = AdditiveEnvelope(
        [
            HarmonicConfinement(coefficient=0.25),
            ElectronElectronCusp(spinless_coefficient=0.25, range_parameter=0.5, eps=0.0),
        ]
    )
    model = SpENNWaveFunction(
        embedding=EmptyEncoder(),
        layers=[nn.Identity()],
        readout=ConstantReadout(),
        envelope=envelope,
    )

    output = model(batch)

    torch.testing.assert_close(output.logabs, envelope(batch))
    torch.testing.assert_close(output.sign, torch.tensor([-1.0, 1.0], dtype=torch.float64))


def test_wavefunction_envelope_shape_must_match_readout_logabs() -> None:
    batch = ElectronBatch(positions=torch.ones(2, 2, 1, dtype=torch.float64))
    model = SpENNWaveFunction(
        embedding=EmptyEncoder(),
        layers=[nn.Identity()],
        readout=ConstantReadout(),
        envelope=BadShapeEnvelope(),
    )

    with pytest.raises(ValueError, match="Envelope output"):
        model(batch)


def test_wavefunction_envelope_must_return_additive_tensor_not_full_output() -> None:
    batch = ElectronBatch(positions=torch.ones(2, 2, 1, dtype=torch.float64))
    model = SpENNWaveFunction(
        embedding=EmptyEncoder(),
        layers=[nn.Identity()],
        readout=ConstantReadout(),
        envelope=FullOutputEnvelope(),
    )

    with pytest.raises(TypeError, match="torch.Tensor"):
        model(batch)


def test_additive_envelope_rejects_malformed_component_output() -> None:
    batch = ElectronBatch(positions=torch.ones(2, 2, 1, dtype=torch.float64))
    envelope = AdditiveEnvelope([FullOutputEnvelope()])

    with pytest.raises(TypeError, match="torch.Tensor"):
        envelope(batch)


def test_spenn_wavefunction_passes_runtime_sign_equivariance_check() -> None:
    batch = ElectronBatch(positions=torch.tensor([[[0.0], [1.0]], [[2.0], [4.0]]], dtype=torch.float64))
    model = SpENNWaveFunction(
        embedding=EmptyEncoder(),
        layers=[nn.Identity()],
        readout=AntisymmetricReadout(),
        envelope=AdditiveEnvelope([HarmonicConfinement(coefficient=0.0)]),
    )

    output = model(batch)

    assert output.validate() is output
    assert_equivariant_all(model, batch)
