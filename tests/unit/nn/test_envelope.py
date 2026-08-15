"""Envelope-factor tests for trainable wavefunction ansatz modules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.permutation import Permutation
from tpen.data.real import Feature
from tpen.nn import (
    AdditiveCusp,
    AdditiveEnvelope,
    AsymptoticDecay,
    ElectronElectronCusp,
    ElectronNucleusCusp,
    ElectronNucleusCuspLaw,
    Envelope,
    GaussianConfinement,
    HookeGaussianConfinement,
    LinearElectronNucleusCuspLaw,
    LogAmplitudeFactor,
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


# --- A4: generic post-readout LogAmplitudeFactor / AdditiveCusp / ElectronNucleusCusp ---


class ConstantLogAmplitudeFactor(LogAmplitudeFactor):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def factor_value(self, batch: ElectronBatch) -> torch.Tensor:
        return torch.full((batch.batch_size,), self.value, device=batch.device, dtype=batch.dtype)


class BadShapeLogAmplitudeFactor(LogAmplitudeFactor):
    def factor_value(self, batch: ElectronBatch) -> torch.Tensor:
        return torch.zeros(batch.batch_size, 1, device=batch.device, dtype=batch.dtype)


class ConstantAsymptoticDecay(AsymptoticDecay):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def decay_value(self, batch: ElectronBatch) -> torch.Tensor:
        return torch.full((batch.batch_size,), self.value, device=batch.device, dtype=batch.dtype)


def test_log_amplitude_factor_base_requires_subclass_implementation() -> None:
    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    with pytest.raises(NotImplementedError):
        LogAmplitudeFactor()(batch)


def test_log_amplitude_factor_rejects_malformed_component_output() -> None:
    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    with pytest.raises(ValueError, match="must have shape"):
        BadShapeLogAmplitudeFactor()(batch)


def test_additive_cusp_sums_component_outputs() -> None:
    batch = ElectronBatch(positions=torch.zeros(3, 2, 1, dtype=torch.float64))
    composed = AdditiveCusp([ConstantLogAmplitudeFactor(0.5), ConstantLogAmplitudeFactor(1.5)])

    torch.testing.assert_close(composed(batch), torch.full((3,), 2.0, dtype=torch.float64))


def test_empty_additive_cusp_returns_zero_batch_vector() -> None:
    batch = ElectronBatch(positions=torch.ones(4, 2, 1, dtype=torch.float64))

    torch.testing.assert_close(AdditiveCusp()(batch), torch.zeros(4, dtype=torch.float64))


def test_additive_cusp_rejects_non_log_amplitude_factor_component() -> None:
    with pytest.raises(TypeError, match="LogAmplitudeFactor"):
        AdditiveCusp([GaussianConfinement(coefficient=0.1)])


def test_additive_cusp_is_itself_a_log_amplitude_factor() -> None:
    composed = AdditiveCusp([ConstantLogAmplitudeFactor(1.0)])

    assert isinstance(composed, LogAmplitudeFactor)


def test_electron_electron_cusp_joins_log_amplitude_factor_interface() -> None:
    cusp = ElectronElectronCusp(spinless_coefficient=0.25, range_parameter=0.5, eps=0.0)

    assert isinstance(cusp, LogAmplitudeFactor)
    composed = AdditiveCusp([cusp])
    batch = ElectronBatch(positions=torch.tensor([[[0.0], [2.0]]], dtype=torch.float64))

    torch.testing.assert_close(composed(batch), cusp(batch))


def test_electron_electron_cusp_state_dict_keys_are_unchanged_by_new_base_class() -> None:
    cusp = ElectronElectronCusp(range_parameter=0.5, trainable_range=True)

    assert set(cusp.state_dict().keys()) == {"raw_same_range", "raw_opposite_range"}


def test_electron_nucleus_cusp_requires_atomic_configuration() -> None:
    with pytest.raises(TypeError, match="AtomicConfiguration"):
        ElectronNucleusCusp(atoms=object())


def test_electron_nucleus_cusp_rejects_law_of_wrong_type() -> None:
    atoms = AtomicConfiguration(
        positions=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        charges=torch.tensor([2.0], dtype=torch.float64),
    )

    with pytest.raises(TypeError, match="ElectronNucleusCuspLaw"):
        ElectronNucleusCusp(atoms=atoms, law=object())


def test_electron_nucleus_cusp_defaults_to_linear_compatibility_law_matching_nuclear_confinement() -> None:
    positions = torch.tensor(
        [[[0.0, 0.0], [1.0e-14, 0.0]], [[3.0, 4.0], [0.0, 2.0]]], dtype=torch.float64
    )
    nuclei = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    charges = torch.tensor([2.0, 1.0], dtype=torch.float64)
    atoms = AtomicConfiguration(positions=nuclei, charges=charges)
    batch = ElectronBatch(positions=positions)

    cusp = ElectronNucleusCusp(atoms=atoms)
    legacy = NuclearConfinement().evaluate(
        ElectronBatch(positions=positions, nuclear_positions=nuclei, nuclear_charges=charges)
    )

    torch.testing.assert_close(cusp(batch), legacy.value.sum(dim=(1, 2)))
    assert isinstance(cusp.law, LinearElectronNucleusCuspLaw)


def test_electron_nucleus_cusp_uses_raw_distance_with_no_clamp() -> None:
    atoms = AtomicConfiguration(
        positions=torch.tensor([[0.0]], dtype=torch.float64),
        charges=torch.tensor([2.0], dtype=torch.float64),
    )
    batch = ElectronBatch(positions=torch.tensor([[[1.0e-14]]], dtype=torch.float64))
    cusp = ElectronNucleusCusp(atoms=atoms)

    # An exactly-coalescent electron-nucleus pair must produce a value whose
    # magnitude tracks the true (unclamped) distance, proving no distance
    # floor was applied.
    assert abs(cusp(batch).item()) < 1.0e-12


def test_electron_nucleus_cusp_satisfies_kato_slope_for_arbitrary_charge() -> None:
    # Independent Kato cusp-condition test: d(value)/dr at r -> 0 must equal
    # -Z, for the linear compatibility law, for an arbitrary (non-He) charge.
    # Isolated to a single nucleus/electron pair, matching the established
    # ElectronElectronCusp slope-test pattern: with more than one pair, the
    # other (non-coalescing) pairs contribute their own generically nonzero
    # slope and would contaminate the measurement.
    tiny_r = torch.tensor(1.0e-7, dtype=torch.float64)
    atoms = AtomicConfiguration(
        positions=torch.tensor([[0.0]], dtype=torch.float64),
        charges=torch.tensor([3.0], dtype=torch.float64),
    )
    cusp = ElectronNucleusCusp(atoms=atoms)
    batch = ElectronBatch(positions=tiny_r.view(1, 1, 1))

    slope = cusp(batch) / tiny_r

    torch.testing.assert_close(slope, torch.tensor([-3.0], dtype=torch.float64), atol=1.0e-6, rtol=0.0)


def test_electron_nucleus_cusp_is_permutation_invariant() -> None:
    atoms = AtomicConfiguration(
        positions=torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float64),
        charges=torch.tensor([1.0, 1.0], dtype=torch.float64),
    )
    cusp = ElectronNucleusCusp(atoms=atoms)
    positions = torch.tensor([[[0.1, 0.2], [0.7, -0.4], [-0.3, 0.9]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    permuted = ElectronBatch(positions=positions[:, [2, 0, 1]])

    torch.testing.assert_close(cusp(batch), cusp(permuted))


def test_electron_nucleus_cusp_composes_with_electron_electron_cusp_via_additive_cusp() -> None:
    atoms = AtomicConfiguration(
        positions=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        charges=torch.tensor([2.0], dtype=torch.float64),
    )
    en_cusp = ElectronNucleusCusp(atoms=atoms)
    ee_cusp = ElectronElectronCusp(spinless_coefficient=0.25, range_parameter=0.5, eps=0.0)
    composed = AdditiveCusp([en_cusp, ee_cusp])
    batch = ElectronBatch(positions=torch.tensor([[[0.5, 0.0], [1.5, 0.5]]], dtype=torch.float64))

    torch.testing.assert_close(composed(batch), en_cusp(batch) + ee_cusp(batch))


class _LiteralElectronNucleusCuspLaw(ElectronNucleusCuspLaw):
    def value(self, distance: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
        return -2.0 * charges * distance


def test_electron_nucleus_cusp_accepts_custom_law() -> None:
    atoms = AtomicConfiguration(
        positions=torch.tensor([[0.0]], dtype=torch.float64),
        charges=torch.tensor([1.0], dtype=torch.float64),
    )
    linear = ElectronNucleusCusp(atoms=atoms)
    doubled = ElectronNucleusCusp(atoms=atoms, law=_LiteralElectronNucleusCuspLaw())
    batch = ElectronBatch(positions=torch.tensor([[[3.0]]], dtype=torch.float64))

    torch.testing.assert_close(doubled(batch), 2.0 * linear(batch))


def test_asymptotic_decay_base_requires_subclass_implementation() -> None:
    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    with pytest.raises(NotImplementedError):
        AsymptoticDecay()(batch)


def test_asymptotic_decay_is_separate_from_cusp_and_envelope_interfaces() -> None:
    decay = ConstantAsymptoticDecay(0.25)

    assert not isinstance(decay, LogAmplitudeFactor)
    assert not isinstance(decay, Envelope)
    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))
    torch.testing.assert_close(decay(batch), torch.full((2,), 0.25, dtype=torch.float64))


def test_additive_envelope_hydra_target_and_constructor_are_unchanged() -> None:
    # Minor-release compatibility: AdditiveEnvelope keeps its exact public
    # identity (import path, constructor signature, ModuleList child name).
    assert AdditiveEnvelope.__module__ == "tpen.nn.envelope"
    envelope = AdditiveEnvelope(envelopes=[GaussianConfinement(coefficient=0.1)], enabled=True)
    assert isinstance(envelope.envelopes, nn.ModuleList)
    assert list(envelope.state_dict().keys()) == []


def test_additive_envelope_state_dict_round_trips_with_trainable_component() -> None:
    envelope = AdditiveEnvelope([GaussianConfinement(coefficient=0.1, trainable=True)])
    state = envelope.state_dict()
    assert set(state.keys()) == {"envelopes.0.raw_coefficient"}

    restored = AdditiveEnvelope([GaussianConfinement(coefficient=0.0, trainable=True)])
    restored.load_state_dict(state)
    batch = ElectronBatch(positions=torch.ones(2, 2, 1, dtype=torch.float64))

    torch.testing.assert_close(restored(batch), envelope(batch))
