"""Envelope-factor tests for trainable wavefunction ansatz modules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch, FactorizedLocalEnergyInput, WavefunctionOutput
from tpen.data.permutation import Permutation
from tpen.data.real import Feature
from tpen.nn import (
    AdditiveCusp,
    AdditiveEnvelope,
    ElectronElectronCusp,
    ElectronNucleusCusp,
    ElectronNucleusCuspLaw,
    Envelope,
    GaussianConfinement,
    HookeGaussianConfinement,
    LinearElectronNucleusCuspLaw,
    LogAmplitudeFactor,
    TPENWaveFunction,
    CurvatureElectronNucleusCuspLaw,
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


def test_wavefunction_allows_no_post_readout_factors() -> None:
    batch = ElectronBatch(positions=torch.tensor([[[0.0], [2.0]]], dtype=torch.float64))
    model = TPENWaveFunction(
        embedding=EmptyEncoder(),
        layers=[nn.Identity()],
        readout=ConstantReadout(),
    )

    output = model(batch)

    torch.testing.assert_close(output.logabs, torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(output.sign, torch.tensor([-1.0], dtype=torch.float64))


def test_wavefunction_composes_legacy_envelope_and_generic_factors_in_one_pipeline() -> None:
    # A5: envelope/nuclear_envelope mutual exclusion is retired -- both a
    # legacy `envelope` and generic post-readout `factors` may be supplied
    # together and sum into one pipeline.
    batch = ElectronBatch(
        positions=torch.tensor([[[0.0], [2.0]], [[1.0], [3.0]]], dtype=torch.float64),
        nuclear_positions=torch.tensor([[0.0]], dtype=torch.float64),
        nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
    )
    atoms = AtomicConfiguration(positions=torch.tensor([[0.0]], dtype=torch.float64), charges=torch.tensor([2.0], dtype=torch.float64))
    envelope = AdditiveEnvelope([GaussianConfinement(coefficient=0.1)])
    en_cusp = ElectronNucleusCusp(atoms=atoms)
    model = TPENWaveFunction(
        embedding=EmptyEncoder(),
        layers=[nn.Identity()],
        readout=ConstantReadout(),
        envelope=envelope,
        factors=[en_cusp],
    )

    output = model(batch)

    torch.testing.assert_close(output.logabs, envelope(batch) + en_cusp(batch))


def test_wavefunction_factors_pipeline_calls_readout_once_and_does_not_alias_aux() -> None:
    batch = ElectronBatch(positions=torch.tensor([[[0.0], [2.0]], [[1.0], [3.0]]], dtype=torch.float64))
    readout = CountingReadout()
    atoms = AtomicConfiguration(positions=torch.tensor([[0.0]], dtype=torch.float64), charges=torch.tensor([2.0], dtype=torch.float64))
    model = TPENWaveFunction(
        embedding=EmptyEncoder(),
        layers=[nn.Identity()],
        readout=readout,
        factors=[ElectronElectronCusp(eps=0.0), ElectronNucleusCusp(atoms=atoms)],
    )

    output = model(batch)

    assert readout.calls == 1
    output.aux["mutation"] = True
    output2 = model(batch)
    assert "mutation" not in output2.aux


def test_factorized_local_energy_input_excludes_bound_cusp_and_reconstructs_full_score() -> None:
    atoms = AtomicConfiguration(
        positions=torch.tensor([[0.0]], dtype=torch.float64),
        charges=torch.tensor([2.0], dtype=torch.float64),
    )
    cusp = ElectronNucleusCusp(atoms, CurvatureElectronNucleusCuspLaw(curvature_coefficient=0.3, curvature_range=1.5))

    class PositionReadout(nn.Module):
        def forward(self, features: Feature, batch: ElectronBatch) -> WavefunctionOutput:
            logabs = batch.positions.square().sum(dim=(1, 2))
            return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))

    positions = torch.tensor([[[0.4]], [[0.8]]], dtype=torch.float64, requires_grad=True)
    batch = ElectronBatch(positions=positions)
    model = TPENWaveFunction(
        embedding=EmptyEncoder(), layers=[nn.Identity()], readout=PositionReadout(), factors=[cusp],
        analytic_cusp_provider=cusp,
    )

    full = model(batch)
    factorized = model.factorized_local_energy_input(batch)
    reconstructed = factorized.regular_wavefunction_output.logabs + factorized.electron_nucleus_cusp_evaluation.pair_value.sum(
        dim=(1, 2)
    )
    torch.testing.assert_close(
        torch.exp(reconstructed - reconstructed[0]),
        torch.exp(full.logabs - full.logabs[0]),
    )
    full_score = torch.autograd.grad(full.logabs.sum(), positions, retain_graph=True)[0]
    regular_score = torch.autograd.grad(factorized.regular_wavefunction_output.logabs.sum(), positions, retain_graph=True)[0]
    cusp_score = torch.autograd.grad(
        factorized.electron_nucleus_cusp_evaluation.pair_value.sum(), positions, retain_graph=True
    )[0]
    torch.testing.assert_close(regular_score + cusp_score, full_score)


def test_factorized_local_energy_input_calls_body_and_provider_exactly_once(monkeypatch) -> None:
    atoms = AtomicConfiguration(positions=torch.zeros(1, 1, dtype=torch.float64), charges=torch.ones(1, dtype=torch.float64))
    cusp = ElectronNucleusCusp(atoms)
    body_calls = 0
    analytic_calls = 0

    class CountingReadout(ConstantReadout):
        def forward(self, features: Feature, batch: ElectronBatch) -> WavefunctionOutput:
            nonlocal body_calls
            body_calls += 1
            return super().forward(features, batch)

    original = cusp.analytic_evaluation

    def analytic_spy(batch: ElectronBatch):
        nonlocal analytic_calls
        analytic_calls += 1
        return original(batch)

    monkeypatch.setattr(cusp, "analytic_evaluation", analytic_spy)
    model = TPENWaveFunction(
        embedding=EmptyEncoder(), layers=[nn.Identity()], readout=CountingReadout(), factors=[cusp],
        analytic_cusp_provider=cusp,
    )

    model.factorized_local_energy_input(ElectronBatch(positions=torch.ones(1, 1, 1, dtype=torch.float64)))

    assert body_calls == 1
    assert analytic_calls == 1


def test_ordinary_forward_never_requests_analytic_cusp_data() -> None:
    atoms = AtomicConfiguration(positions=torch.zeros(1, 1, dtype=torch.float64), charges=torch.ones(1, dtype=torch.float64))
    cusp = ElectronNucleusCusp(atoms)
    calls = 0
    original = cusp.analytic_evaluation

    def analytic_spy(batch: ElectronBatch):
        nonlocal calls
        calls += 1
        return original(batch)

    cusp.analytic_evaluation = analytic_spy
    model = TPENWaveFunction(
        embedding=EmptyEncoder(), layers=[nn.Identity()], readout=ConstantReadout(), factors=[cusp],
        analytic_cusp_provider=cusp,
    )
    model(ElectronBatch(positions=torch.ones(1, 1, 1, dtype=torch.float64)))
    assert calls == 0


def test_analytic_provider_binding_rejects_missing_or_duplicate_identity() -> None:
    atoms = AtomicConfiguration(positions=torch.zeros(1, 1, dtype=torch.float64), charges=torch.ones(1, dtype=torch.float64))
    cusp = ElectronNucleusCusp(atoms)
    other = ElectronNucleusCusp(atoms)
    kwargs = dict(embedding=EmptyEncoder(), layers=[nn.Identity()], readout=ConstantReadout())

    with pytest.raises(ValueError, match="unique participating"):
        TPENWaveFunction(**kwargs, factors=[cusp, cusp], analytic_cusp_provider=cusp)
    with pytest.raises(ValueError, match="unique participating"):
        TPENWaveFunction(**kwargs, factors=[other], analytic_cusp_provider=cusp)

    model = TPENWaveFunction(**kwargs, factors=[cusp])
    with pytest.raises(ValueError, match="requires an analytic_cusp_provider"):
        model.factorized_local_energy_input(ElectronBatch(positions=torch.ones(1, 1, 1, dtype=torch.float64)))


def test_factorized_local_energy_input_typed_contracts_cover_both_components() -> None:
    atoms = AtomicConfiguration(positions=torch.zeros(1, 1, dtype=torch.float64), charges=torch.ones(1, dtype=torch.float64))
    cusp = ElectronNucleusCusp(atoms)
    model = TPENWaveFunction(
        embedding=EmptyEncoder(), layers=[nn.Identity()], readout=ConstantReadout(), factors=[cusp],
        analytic_cusp_provider=cusp,
    )
    value = model.factorized_local_energy_input(ElectronBatch(positions=torch.ones(1, 2, 1, dtype=torch.float64)))
    round_trip = value.permute(Permutation((1, 0))).permute(Permutation((1, 0)))
    close, metrics = value.compare(round_trip)
    assert close
    assert metrics["max_abs_error"] == pytest.approx(0.0)
    assert isinstance(value, FactorizedLocalEnergyInput)


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


def test_electron_nucleus_cusp_defaults_to_linear_compatibility_law_matching_legacy_formula() -> None:
    # The linear compatibility law must reproduce the He linear cusp formerly
    # hard-coded by the retired `NuclearConfinement` envelope: `-Z * r`,
    # summed over electron-nucleus pairs.
    positions = torch.tensor(
        [[[0.0, 0.0], [1.0e-14, 0.0]], [[3.0, 4.0], [0.0, 2.0]]], dtype=torch.float64
    )
    nuclei = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    charges = torch.tensor([2.0, 1.0], dtype=torch.float64)
    atoms = AtomicConfiguration(positions=nuclei, charges=charges)
    batch = ElectronBatch(positions=positions)

    cusp = ElectronNucleusCusp(atoms=atoms)
    expected_distance = torch.linalg.vector_norm(positions.unsqueeze(2) - nuclei.view(1, 1, 2, 2), dim=-1)
    expected_value = (-expected_distance * charges.view(1, 1, 2)).sum(dim=(1, 2))

    torch.testing.assert_close(cusp(batch), expected_value)
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


# --- A6: generic N=2 (H2) ElectronNucleusCusp data coverage ---


def _h2_atoms(dtype: torch.dtype = torch.float64) -> AtomicConfiguration:
    return AtomicConfiguration(
        positions=torch.tensor([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=dtype),
        charges=torch.tensor([1.0, 1.0], dtype=dtype),
    )


def test_electron_nucleus_cusp_satisfies_kato_slope_per_nucleus_for_h2() -> None:
    # H2's two nuclei each independently satisfy the Kato cusp condition. The
    # far nucleus sits off the coalescence axis (a perpendicular offset), so
    # its own distance changes only to second order as the electron
    # approaches the near nucleus; subtracting the exact-coalescence value
    # before dividing by tiny_r removes that residual background instead of
    # merely hoping it is small, isolating the near nucleus's own -Z slope.
    tiny_r = torch.tensor(1.0e-7, dtype=torch.float64)
    offset = 1.0
    for near_index, far_index in ((0, 1), (1, 0)):
        positions = torch.zeros(2, 3, dtype=torch.float64)
        positions[far_index] = torch.tensor([offset, 0.0, 0.0], dtype=torch.float64)
        charges = torch.tensor([1.0, 1.0], dtype=torch.float64)
        charges[near_index] = 3.0
        atoms = AtomicConfiguration(positions=positions, charges=charges)
        cusp = ElectronNucleusCusp(atoms=atoms)
        near_position = positions[near_index]
        coalescent_batch = ElectronBatch(positions=near_position.clone().view(1, 1, 3))
        displaced_position = near_position + torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) * tiny_r
        displaced_batch = ElectronBatch(positions=displaced_position.view(1, 1, 3))

        slope = (cusp(displaced_batch) - cusp(coalescent_batch)) / tiny_r

        torch.testing.assert_close(slope, torch.tensor([-3.0], dtype=torch.float64), atol=1.0e-6, rtol=0.0)


def test_electron_nucleus_cusp_is_nucleus_relabel_invariant_for_h2() -> None:
    atoms = _h2_atoms()
    relabeled = AtomicConfiguration(positions=atoms.positions.flip(0), charges=atoms.charges.flip(0))
    cusp = ElectronNucleusCusp(atoms=atoms)
    relabeled_cusp = ElectronNucleusCusp(atoms=relabeled)
    batch = ElectronBatch(positions=torch.tensor([[[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]]], dtype=torch.float64))

    torch.testing.assert_close(cusp(batch), relabeled_cusp(batch))


def test_electron_nucleus_cusp_h2_raw_exact_zero_boundary_at_one_nucleus() -> None:
    # An electron exactly coincident with one of H2's two nuclei must produce
    # a cusp value equal to the other nucleus's raw (unclamped) contribution
    # alone -- proving no distance floor is applied at coalescence even with
    # more than one nucleus present.
    atoms = _h2_atoms()
    cusp = ElectronNucleusCusp(atoms=atoms)
    batch = ElectronBatch(positions=atoms.positions[0].clone().view(1, 1, 3))

    distance_to_far_nucleus = torch.linalg.norm(atoms.positions[0] - atoms.positions[1])
    expected = -atoms.charges[1] * distance_to_far_nucleus
    torch.testing.assert_close(cusp(batch), expected.view(1))


def test_electron_nucleus_cusp_respects_dtype_and_device_for_h2() -> None:
    atoms = _h2_atoms(dtype=torch.float32)
    cusp = ElectronNucleusCusp(atoms=atoms)
    positions = torch.tensor([[[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]]], dtype=torch.float32)
    batch = ElectronBatch(positions=positions)

    output = cusp(batch)

    assert output.dtype is torch.float32
    assert output.device == positions.device


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


def test_trainable_curvature_law_preserves_charge_fixed_kato_slope() -> None:
    # The curvature term must contribute only at second order, so the
    # first-order Kato slope stays exactly -Z regardless of curvature params.
    tiny_r = torch.tensor(1.0e-7, dtype=torch.float64)
    atoms = AtomicConfiguration(
        positions=torch.tensor([[0.0]], dtype=torch.float64),
        charges=torch.tensor([3.0], dtype=torch.float64),
    )
    law = CurvatureElectronNucleusCuspLaw(curvature_coefficient=5.0, curvature_range=2.0)
    cusp = ElectronNucleusCusp(atoms=atoms, law=law)
    batch = ElectronBatch(positions=tiny_r.view(1, 1, 1))

    slope = cusp(batch) / tiny_r

    torch.testing.assert_close(slope, torch.tensor([-3.0], dtype=torch.float64), atol=1.0e-6, rtol=0.0)


def test_trainable_curvature_law_is_second_order_and_trainable() -> None:
    # The curvature contribution w_A(r) = c r^2 / (1 + d r) must match its
    # analytic value away from coalescence, and both c and d must be
    # differentiable when trainable.
    distance = torch.tensor(0.5, dtype=torch.float64)
    charges = torch.tensor(2.0, dtype=torch.float64)
    law = CurvatureElectronNucleusCuspLaw(curvature_coefficient=0.3, curvature_range=1.5)

    value = law.value(distance, charges)
    expected_linear = -charges * distance
    expected_curvature = 0.3 * distance.square() / (1.0 + 1.5 * distance)
    torch.testing.assert_close(value, expected_linear + expected_curvature)

    value.backward()
    assert law.raw_curvature_coefficient.grad is not None
    assert law.raw_curvature_range.grad is not None
    assert law.curvature_range > 0.0


def test_trainable_curvature_law_zero_coefficient_matches_linear_law() -> None:
    atoms = AtomicConfiguration(
        positions=torch.tensor([[0.0]], dtype=torch.float64),
        charges=torch.tensor([2.0], dtype=torch.float64),
    )
    linear = ElectronNucleusCusp(atoms=atoms, law=LinearElectronNucleusCuspLaw())
    curved = ElectronNucleusCusp(
        atoms=atoms,
        law=CurvatureElectronNucleusCuspLaw(curvature_coefficient=0.0, curvature_range=1.0, trainable=False),
    )
    batch = ElectronBatch(positions=torch.tensor([[[0.7]]], dtype=torch.float64))

    torch.testing.assert_close(curved(batch), linear(batch))


def test_trainable_curvature_law_rejects_nonpositive_range() -> None:
    with pytest.raises(ValueError, match="curvature_range"):
        CurvatureElectronNucleusCuspLaw(curvature_range=0.0)


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


def test_feature_envelope_name_is_reserved_and_unused() -> None:
    # C0 terminology contract: `FeatureEnvelope` is reserved for a future
    # typed feature-space transform and must never exist as an alias/rename
    # of the current multiplicative `Envelope`.
    import tpen.nn as tpen_nn
    import tpen.nn.envelope as envelope_module

    assert not hasattr(envelope_module, "FeatureEnvelope")
    assert not hasattr(tpen_nn, "FeatureEnvelope")


def test_envelope_module_docstring_documents_compatibility_terminology() -> None:
    # C0 terminology contract: the module docstring is the load-bearing
    # source for the Envelope/LogAmplitudeFactor compatibility split; pin its
    # key claims so a future edit cannot silently drop them.
    import tpen.nn.envelope as envelope_module

    doc = envelope_module.__doc__
    assert "compatibility surface" in doc
    assert "runtime deprecation warning" in doc
    assert "FeatureEnvelope" in doc
    assert "TPENWaveFunction.factors" in doc


def test_electron_nucleus_cusp_law_documents_regular_component_contract() -> None:
    # C2 terminology contract: the optional trainable regular radial
    # component for second-order curvature is documented on the base law.
    doc = ElectronNucleusCuspLaw.__doc__
    assert "second-order" in doc
    assert "w_A" in doc
    assert "CurvatureElectronNucleusCuspLaw" in doc
    # DISCRIMINATING, NOT MERELY SATISFIABLE. The assertion above cannot fail on
    # the property this rename exists to establish. The retired name was exactly
    # this one with a `Trainable` prefix, so that assertion reduces to
    #     NEW in "Trainable" + NEW  ->  trivially True
    # and it passes on an un-renamed docstring. Pinning the prefix's ABSENCE is
    # what lets the pair fail. Scoped to this base-class docstring on purpose:
    # `CurvatureElectronNucleusCuspLaw.__doc__` deliberately retains one prose
    # mention of the former name so the rename stays traceable, so a repo-wide
    # absence check would be wrong here.
    assert "TrainableCurvature" not in doc


# --- H-R6: trainable range parameter in the electron-nucleus cusp ---


def _single_nucleus(charge: float) -> AtomicConfiguration:
    """Return a one-dimensional single-nucleus configuration of the given charge."""

    return AtomicConfiguration(
        positions=torch.tensor([[0.0]], dtype=torch.float64),
        charges=torch.tensor([charge], dtype=torch.float64),
    )


def test_trainable_curvature_kato_slope_converges_to_minus_charge_as_r_goes_to_zero() -> None:
    # H-R6 requirement 1: prove the LIMIT d/dr v_A(r) -> -Z as r -> 0, rather
    # than asserting it at a single small radius. The error must shrink with r
    # and the slope must be exactly -Z at coalescence, for hostile curvature
    # parameters that dominate the value at ordinary radii.
    charge = 3.0
    charges = torch.tensor(charge, dtype=torch.float64)
    law = CurvatureElectronNucleusCuspLaw(curvature_coefficient=7.0, curvature_range=0.5)

    errors = []
    for radius in (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6):
        distance = torch.tensor(radius, dtype=torch.float64, requires_grad=True)
        (slope,) = torch.autograd.grad(law.value(distance, charges), distance)
        errors.append(abs(slope.item() + charge))

    # Strictly decreasing error, converging to zero: this is the limit, not a
    # single lucky sample.
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1.0e-4
    assert errors[0] > errors[-1] * 100.0

    at_origin = torch.zeros((), dtype=torch.float64, requires_grad=True)
    (slope_at_origin,) = torch.autograd.grad(law.value(at_origin, charges), at_origin)
    torch.testing.assert_close(slope_at_origin, torch.tensor(-charge, dtype=torch.float64))


def test_trainable_curvature_range_is_reachable_from_wavefunction_parameters() -> None:
    # H-R6 requirement 2: a "trainable" range absent from model.parameters() is
    # the silent failure this slice exists to prevent. Assert registration
    # through the real TPENWaveFunction factors pipeline AND that a backward
    # pass through the model populates a nonzero gradient on it.
    law = CurvatureElectronNucleusCuspLaw(curvature_coefficient=0.4, curvature_range=1.25)
    cusp = ElectronNucleusCusp(atoms=_single_nucleus(2.0), law=law)
    model = TPENWaveFunction(
        embedding=EmptyEncoder(),
        layers=[nn.Identity()],
        readout=ConstantReadout(),
        factors=[cusp],
    )

    names = dict(model.named_parameters())
    assert "factors.0.law.raw_curvature_range" in names
    assert "factors.0.law.raw_curvature_coefficient" in names
    # Identity, not just name equality: the optimizer must see this exact tensor.
    assert any(parameter is law.raw_curvature_range for parameter in model.parameters())

    batch = ElectronBatch(
        positions=torch.tensor([[[0.6], [1.1]], [[1.4], [0.3]]], dtype=torch.float64)
    )
    model(batch).logabs.sum().backward()

    assert law.raw_curvature_range.grad is not None
    assert law.raw_curvature_range.grad.item() != 0.0


def test_trainable_curvature_range_stays_positive_under_hostile_update() -> None:
    # H-R6 requirement 3: the softplus reparameterization must keep the range
    # strictly positive even after an adverse step far larger than any real one.
    law = CurvatureElectronNucleusCuspLaw(curvature_coefficient=0.4, curvature_range=1.0)
    with torch.no_grad():
        law.raw_curvature_range.sub_(1.0e6)

    range_parameter = law.curvature_range
    assert torch.isfinite(range_parameter)
    assert range_parameter.item() > 0.0

    distance = torch.tensor([0.0, 1.0e-3, 2.0, 1.0e3], dtype=torch.float64)
    value = law.value(distance, torch.tensor(2.0, dtype=torch.float64))
    assert torch.isfinite(value).all()


def test_trainable_curvature_outer_tail_slope_is_shifted_and_does_not_saturate() -> None:
    # H-R6 recorded functional-form decision, asserted rather than assumed:
    # v_A(r) = -Z r + c r^2 / (1 + d r) has outer-tail slope -Z + c/d and keeps
    # growing linearly, unlike the saturating Pade law -Z r / (1 + a r).
    coefficient, range_parameter, charge = 0.5, 2.0, 3.0
    charges = torch.tensor(charge, dtype=torch.float64)
    law = CurvatureElectronNucleusCuspLaw(
        curvature_coefficient=coefficient, curvature_range=range_parameter
    )
    expected = -charge + coefficient / range_parameter

    torch.testing.assert_close(
        law.outer_tail_slope(charges), torch.tensor(expected, dtype=torch.float64)
    )
    assert expected != -charge  # the tail slope really is shifted, not incidental

    for radius, tolerance in ((1.0e3, 1.0e-3), (1.0e5, 1.0e-5)):
        distance = torch.tensor(radius, dtype=torch.float64, requires_grad=True)
        (slope,) = torch.autograd.grad(law.value(distance, charges), distance)
        assert abs(slope.item() - expected) < tolerance

    # Non-saturating: doubling r doubles the value.
    far = torch.tensor([1.0e4, 2.0e4], dtype=torch.float64)
    values = law.value(far, charges)
    assert (values[1] / values[0]).item() > 1.9

    # Contrast, made executable: the saturating Pade law satisfies the same
    # Kato slope but stops growing, so log|psi| stops decaying.
    saturating = -charge * far / (1.0 + 0.5 * far)
    assert abs((saturating[1] / saturating[0]).item() - 1.0) < 0.01


def test_trainable_curvature_zero_coefficient_delays_range_gradient_by_one_step() -> None:
    # Documented degeneracy: at exactly c = 0 the range gradient is identically
    # zero, but c itself still moves and unlocks the range from the next step.
    # A config that wants the range trained from step one must init c nonzero.
    law = CurvatureElectronNucleusCuspLaw(curvature_coefficient=0.0, curvature_range=1.0)
    distance = torch.tensor(0.8, dtype=torch.float64)
    charges = torch.tensor(2.0, dtype=torch.float64)

    law.value(distance, charges).backward()
    assert law.raw_curvature_range.grad.item() == 0.0
    assert law.raw_curvature_coefficient.grad.item() != 0.0

    with torch.no_grad():
        law.raw_curvature_coefficient.add_(0.1)
    law.zero_grad(set_to_none=True)
    law.value(distance, charges).backward()

    assert law.raw_curvature_range.grad.item() != 0.0


def test_trainable_curvature_changes_checkpoint_state_and_blocks_strict_cross_restore() -> None:
    # Checkpoint consequence stated in the H-R6 receipt, made executable: the
    # default (linear) and fixed-curvature cusps carry NO state, the trainable
    # law adds two keys, and a strict=True restore across the two shapes fails
    # loudly rather than silently dropping the trainable range.
    atoms = _single_nucleus(2.0)
    default_cusp = ElectronNucleusCusp(atoms=atoms)
    fixed_cusp = ElectronNucleusCusp(
        atoms=atoms,
        law=CurvatureElectronNucleusCuspLaw(
            curvature_coefficient=0.4, curvature_range=1.25, trainable=False
        ),
    )
    trainable_cusp = ElectronNucleusCusp(
        atoms=atoms,
        law=CurvatureElectronNucleusCuspLaw(curvature_coefficient=0.4, curvature_range=1.25),
    )

    assert list(default_cusp.state_dict().keys()) == []
    assert list(fixed_cusp.state_dict().keys()) == []
    assert set(trainable_cusp.state_dict().keys()) == {
        "law.raw_curvature_coefficient",
        "law.raw_curvature_range",
    }

    batch = ElectronBatch(positions=torch.tensor([[[0.7]], [[1.9]]], dtype=torch.float64))
    restored = ElectronNucleusCusp(
        atoms=atoms,
        law=CurvatureElectronNucleusCuspLaw(curvature_coefficient=1.0, curvature_range=3.0),
    )
    restored.load_state_dict(trainable_cusp.state_dict(), strict=True)
    torch.testing.assert_close(restored(batch), trainable_cusp(batch))

    with pytest.raises(RuntimeError):
        ElectronNucleusCusp(atoms=atoms).load_state_dict(trainable_cusp.state_dict(), strict=True)


def test_trainable_curvature_law_documents_outer_tail_consequence() -> None:
    # The functional-form decision is load-bearing for H-R4 tail tolerances;
    # pin its key claims so a future edit cannot silently drop them.
    doc = CurvatureElectronNucleusCuspLaw.__doc__
    assert "-Z_A + c / d" in doc
    assert "NON-saturating" in doc
    assert "must not be applied unchanged" in doc
    assert "range parameter" in doc
