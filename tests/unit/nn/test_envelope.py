"""Envelope-factor tests for trainable wavefunction ansatz modules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.permutation import Permutation
from tpen.data.real import Feature
from tpen.nn import (
    AdditiveEnvelope,
    ElectronElectronCusp,
    Envelope,
    GaussianConfinement,
    HookeGaussianConfinement,
    NuclearConfinement,
    NuclearFactorizedEnvelope,
    TPENWaveFunction,
)
from tests.helpers.equivariance import assert_equivariant_all
from tests.helpers.hooke_models import build_tiny_spenn


class EmptyEncoder(nn.Module):
    def forward(self, batch: ElectronBatch, *, context=None) -> Feature:
        return Feature()


class ConstantReadout(nn.Module):
    def forward(self, features: Feature, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = torch.zeros(batch.batch_size, device=batch.device, dtype=batch.dtype)
        sign = torch.tensor([-1.0, 1.0], device=batch.device, dtype=batch.dtype)[: batch.batch_size]
        return WavefunctionOutput(logabs=logabs, sign=sign)


class CountingReadout(ConstantReadout):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, features: Feature, batch: ElectronBatch) -> WavefunctionOutput:
        self.calls += 1
        return super().forward(features, batch)


class AntisymmetricReadout(nn.Module):
    def forward(self, features: Feature, batch: ElectronBatch) -> WavefunctionOutput:
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
    envelope = GaussianConfinement(coefficient=0.25)

    values = envelope(batch)

    expected = -0.25 * positions.square().sum(dim=(1, 2))
    torch.testing.assert_close(values, expected)


def test_harmonic_confinement_is_permutation_invariant() -> None:
    positions = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    permuted = ElectronBatch(positions=positions[:, [2, 0, 1]])
    envelope = GaussianConfinement(coefficient=0.25)

    torch.testing.assert_close(envelope(batch), envelope(permuted))


def test_harmonic_confinement_trainable_coefficient_is_nonnegative_and_differentiable() -> None:
    positions = torch.tensor([[[1.0], [2.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    envelope = GaussianConfinement(coefficient=0.25, trainable=True)

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

    values = GaussianConfinement(enabled=False, coefficient=0.25)(batch)

    torch.testing.assert_close(values, torch.zeros(4, dtype=torch.float64))


def test_additive_envelope_sums_component_outputs() -> None:
    positions = torch.tensor([[[0.0], [2.0]], [[1.0], [4.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    harmonic = GaussianConfinement(coefficient=0.25)
    cusp = ElectronElectronCusp(spinless_coefficient=0.25, range_parameter=0.5, eps=0.0)
    envelope = AdditiveEnvelope([harmonic, cusp])

    torch.testing.assert_close(envelope(batch), harmonic(batch) + cusp(batch))


def test_empty_additive_envelope_returns_zero_batch_vector() -> None:
    batch = ElectronBatch(positions=torch.ones(4, 2, 3, dtype=torch.float64))

    values = AdditiveEnvelope()(batch)

    torch.testing.assert_close(values, torch.zeros(4, dtype=torch.float64))


def test_nuclear_confinement_exposes_raw_he_radial_factorization() -> None:
    positions = torch.tensor(
        [[[0.0, 0.0], [1.0e-14, 0.0]], [[3.0, 4.0], [0.0, 2.0]]], dtype=torch.float64
    )
    nuclei = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    charges = torch.tensor([2.0, 1.0], dtype=torch.float64)
    batch = ElectronBatch(positions=positions, nuclear_positions=nuclei, nuclear_charges=charges)

    evaluation = NuclearConfinement().evaluate(batch)

    expected_distance = torch.linalg.vector_norm(positions.unsqueeze(2) - nuclei.view(1, 1, 2, 2), dim=-1)
    torch.testing.assert_close(evaluation.distance, expected_distance)
    assert evaluation.distance[0, 1, 0] < 1.0e-12  # Proves no clamped-distance helper was used.
    torch.testing.assert_close(evaluation.value, -expected_distance * charges.view(1, 1, 2))
    torch.testing.assert_close(evaluation.radial_first_derivative, -charges.view(1, 1, 2).expand_as(expected_distance))
    torch.testing.assert_close(evaluation.radial_second_derivative, torch.zeros_like(expected_distance))
    torch.testing.assert_close(evaluation.origin_radial_derivative, -charges.view(1, 2).expand(2, 2))
    assert evaluation.validate(batch) is evaluation


def test_nuclear_confinement_typed_permutation_leaves_origin_derivative_fixed() -> None:
    batch = ElectronBatch(
        positions=torch.tensor([[[0.0], [2.0]]], dtype=torch.float64),
        nuclear_positions=torch.tensor([[0.0]], dtype=torch.float64),
        nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
    )
    evaluation = NuclearConfinement().evaluate(batch)
    permuted = evaluation.permute(Permutation((1, 0)))

    torch.testing.assert_close(permuted.distance, evaluation.distance[:, [1, 0]])
    torch.testing.assert_close(permuted.origin_radial_derivative, evaluation.origin_radial_derivative)
    close, metrics = evaluation.compare(permuted.permute(Permutation((1, 0))))
    assert close, metrics


def test_factorized_nuclear_wavefunction_keeps_atom_ownership_explicit() -> None:
    batch = ElectronBatch(
        positions=torch.tensor([[[0.0], [2.0]], [[1.0], [3.0]]], dtype=torch.float64),
        nuclear_positions=torch.tensor([[0.0]], dtype=torch.float64),
        nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
    )
    factorized = NuclearFactorizedEnvelope(AdditiveEnvelope(), NuclearConfinement())
    model = TPENWaveFunction(
        embedding=EmptyEncoder(),
        layers=[nn.Identity()],
        readout=ConstantReadout(),
        nuclear_envelope=factorized,
    )

    parts = model.nuclear_factorization(batch)
    output = parts.as_output()
    output.aux["mutation"] = True
    assert "mutation" not in parts.aux
    assert parts.validate(batch) is parts
    with pytest.raises(ValueError, match="exactly one"):
        TPENWaveFunction(
            embedding=EmptyEncoder(),
            layers=[nn.Identity()],
            readout=ConstantReadout(),
            envelope=AdditiveEnvelope(),
            nuclear_envelope=factorized,
        )

    legacy = TPENWaveFunction(
        embedding=EmptyEncoder(),
        layers=[nn.Identity()],
        readout=ConstantReadout(),
        envelope=AdditiveEnvelope(),
    )
    with pytest.raises(ValueError, match="no NuclearFactorizedEnvelope"):
        legacy.nuclear_factorization(batch)


def test_factorized_nuclear_wavefunction_constructs_one_readout_per_top_level_call() -> None:
    batch = ElectronBatch(
        positions=torch.tensor([[[0.0], [2.0]], [[1.0], [3.0]]], dtype=torch.float64),
        nuclear_positions=torch.tensor([[0.0]], dtype=torch.float64),
        nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
    )
    readout = CountingReadout()
    model = TPENWaveFunction(
        embedding=EmptyEncoder(),
        layers=[nn.Identity()],
        readout=readout,
        nuclear_envelope=NuclearFactorizedEnvelope(AdditiveEnvelope(), NuclearConfinement()),
    )

    parts = model.nuclear_factorization(batch)
    assert readout.calls == 1
    output = model(batch)
    assert readout.calls == 2
    assert output.validate(batch_size=batch.batch_size) is output
    assert parts.nuclear.validate(batch) is parts.nuclear


def test_wavefunction_requires_envelope() -> None:
    with pytest.raises(ValueError, match="envelope"):
        TPENWaveFunction(
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
            GaussianConfinement(coefficient=0.25),
            ElectronElectronCusp(spinless_coefficient=0.25, range_parameter=0.5, eps=0.0),
        ]
    )
    model = TPENWaveFunction(
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
    model = TPENWaveFunction(
        embedding=EmptyEncoder(),
        layers=[nn.Identity()],
        readout=ConstantReadout(),
        envelope=BadShapeEnvelope(),
    )

    with pytest.raises(ValueError, match="Envelope output"):
        model(batch)


def test_wavefunction_envelope_must_return_additive_tensor_not_full_output() -> None:
    batch = ElectronBatch(positions=torch.ones(2, 2, 1, dtype=torch.float64))
    model = TPENWaveFunction(
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
    model = TPENWaveFunction(
        embedding=EmptyEncoder(),
        layers=[nn.Identity()],
        readout=AntisymmetricReadout(),
        envelope=AdditiveEnvelope([GaussianConfinement(coefficient=0.0)]),
    )

    output = model(batch)

    assert output.validate() is output
    assert_equivariant_all(model, batch)


def test_additive_envelope_composes_cusp_and_hooke_gaussian_exactly() -> None:
    # T8: the composed wavefunction-level envelope stack (revised D5) must
    # reproduce the sum of its parts exactly.
    positions = torch.tensor(
        [[[0.2, -0.1, 0.4], [0.7, 0.3, -0.6]], [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=torch.float64
    )
    spins = torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions, spins=spins)
    cusp = ElectronElectronCusp(range_parameter=0.5, eps=0.0)
    confinement = HookeGaussianConfinement(omega=0.5)
    envelope = AdditiveEnvelope([cusp, confinement])

    torch.testing.assert_close(envelope(batch), cusp(batch) + confinement(batch), rtol=0.0, atol=0.0)


def test_composed_cusp_confinement_envelope_is_permutation_invariant() -> None:
    # T8: the whole envelope stack must stay symmetric under particle
    # exchange so the readout keeps sole ownership of antisymmetry.
    positions = torch.tensor([[[0.1, -0.2, 0.3], [0.7, 0.4, -0.5], [-0.6, 0.2, 0.9]]], dtype=torch.float64)
    spins = torch.tensor([[1.0, -1.0, 1.0]], dtype=torch.float64)
    envelope = AdditiveEnvelope([ElectronElectronCusp(eps=0.0), HookeGaussianConfinement(omega=0.5)])
    batch = ElectronBatch(positions=positions, spins=spins)
    permuted = ElectronBatch(positions=positions[:, [2, 0, 1]], spins=spins[:, [2, 0, 1]])

    torch.testing.assert_close(envelope(batch), envelope(permuted))


def test_wavefunction_logabs_decays_along_radial_rays_beyond_documented_radius() -> None:
    # T8 decay assertion (new diagnostic): with the confinement term enabled,
    # log|psi| of the full tiny Hooke pair model must decrease monotonically
    # along radial rays beyond the documented radius r >= 4. Beyond it the
    # Gaussian confinement (-0.25 * r^2 at omega = 0.5) dominates the
    # polynomial/logarithmic growth of the network readout and the bounded
    # cusp term, so strict monotone decay is architecture-guaranteed.
    torch.manual_seed(0)
    model = build_tiny_spenn()
    generator = torch.Generator().manual_seed(7)
    direction = torch.randn(1, 2, 3, generator=generator, dtype=torch.float64)
    # Normalize the configuration so sum_i |r_i|^2 == 1; the ray parameter is
    # then exactly the configuration radius sqrt(sum_i |r_i|^2).
    direction = direction / direction.square().sum().sqrt()
    radii = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0, 10.0], dtype=torch.float64)
    positions = radii.reshape(-1, 1, 1) * direction
    spins = torch.tensor([[1.0, -1.0]], dtype=torch.float64).expand(radii.shape[0], -1)
    batch = ElectronBatch(positions=positions, spins=spins)

    logabs = model(batch).logabs

    assert torch.all(torch.isfinite(logabs))
    differences = logabs[1:] - logabs[:-1]
    assert torch.all(differences < 0), f"log|psi| must decay along the radial ray beyond r=4, got {logabs.tolist()}"
