"""Physics and local-energy sanity tests."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.nn.cusp import ElectronElectronCusp
from spenn.physics.hamiltonian import (
    LocalEnergyResult,
    local_energy,
    normalize_hamiltonian_terms,
    reference_energy_metrics,
    summarize_local_energy,
)
from spenn.physics.hooke import HookeSingletExact, HookeTripletExact
from spenn.physics.kinetic import KineticEnergy, kinetic_energy_from_logabs
from spenn.physics.potential import (
    ElectronElectronInteraction,
    ElectronNucleusInteraction,
    HarmonicTrap,
)


class GaussianOutputModel(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float64)

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = -self.alpha * batch.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class TrainableGaussianOutputModel(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float64))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = -self.alpha * batch.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class CuspGaussianOutputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
        self.cusp = ElectronElectronCusp(
            same_spin_coefficient=0.25,
            opposite_spin_coefficient=0.5,
            range_parameter=0.5,
            eps=1.0e-12,
        )

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = self.cusp(batch) - self.alpha * batch.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class _ConstantTerm:
    """Hamiltonian term returning fixed values, for helper tests."""

    def __init__(self, name: str, value: torch.Tensor) -> None:
        self.name = name
        self._value = value

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        return LocalEnergyResult(total=self._value, terms={self.name: self._value})


def test_potential_terms_match_direct_hand_calculations() -> None:
    positions = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.5, -1.0], [2.5, 1.0]],
        ],
        dtype=torch.float64,
    )
    nuclei = torch.tensor([[0.0, -1.0], [2.0, 0.0]], dtype=torch.float64)
    charges = torch.tensor([2.0, 0.5], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)

    harmonic = HarmonicTrap(omega=1.5).local_energy(None, batch).total
    repulsion = ElectronElectronInteraction().local_energy(None, batch).total
    attraction = ElectronNucleusInteraction(nuclei, charges).local_energy(None, batch).total

    expected_harmonic = 0.5 * (1.5**2) * positions.square().sum(dim=(1, 2))
    expected_repulsion = torch.linalg.norm(positions[:, 0] - positions[:, 1], dim=-1).reciprocal()
    expected_attraction = -(
        charges.view(1, 1, -1)
        / torch.linalg.norm(positions.unsqueeze(2) - nuclei.view(1, 1, 2, 2), dim=-1)
    ).sum(dim=(1, 2))

    assert torch.allclose(harmonic, expected_harmonic)
    assert torch.allclose(repulsion, expected_repulsion)
    assert torch.allclose(attraction, expected_attraction)


def test_autograd_kinetic_matches_gaussian_logabs_formula() -> None:
    positions = torch.tensor(
        [
            [[1.0, 2.0], [0.5, -1.0]],
            [[-1.5, 0.25], [2.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    alpha = 0.3
    batch = ElectronBatch(positions=positions)

    kinetic = kinetic_energy_from_logabs(GaussianOutputModel(alpha), batch)

    n_electrons = positions.shape[1]
    spatial_dim = positions.shape[2]
    expected = alpha * n_electrons * spatial_dim - 2.0 * alpha**2 * positions.square().sum(dim=(1, 2))
    assert torch.allclose(kinetic, expected)


def test_harmonic_oscillator_ground_state_has_constant_local_energy() -> None:
    omega = 1.7
    positions = torch.tensor(
        [
            [[1.0, 0.0, -1.0], [0.5, 2.0, 0.25]],
            [[-0.5, 1.5, 2.5], [1.0, -2.0, 0.5]],
        ],
        dtype=torch.float64,
    )
    batch = ElectronBatch(positions=positions)
    terms = [KineticEnergy(), HarmonicTrap(omega=omega)]

    energy = local_energy(terms, GaussianOutputModel(alpha=omega / 2.0), batch)

    n_electrons, spatial_dim = positions.shape[1], positions.shape[2]
    expected = torch.full((2,), n_electrons * spatial_dim * omega / 2.0, dtype=torch.float64)
    assert torch.allclose(energy, expected)


def test_local_energy_accepts_wavefunction_output_and_preserves_parameter_gradients() -> None:
    batch = ElectronBatch(
        positions=torch.tensor([[[1.0, 0.0], [0.0, 2.0]], [[-1.0, 1.0], [2.0, -0.5]]], dtype=torch.float64),
    )
    model = TrainableGaussianOutputModel(alpha=0.25)
    terms = [KineticEnergy(), HarmonicTrap(omega=1.0)]

    energy = local_energy(terms, model, batch)
    energy.mean().backward()

    assert energy.shape == (2,)
    assert torch.all(torch.isfinite(energy))
    assert model.alpha.grad is not None
    assert torch.isfinite(model.alpha.grad)


def test_cusp_local_energy_has_finite_second_derivatives_with_pair_diagonal() -> None:
    batch = ElectronBatch(
        positions=torch.tensor([[[0.25, -0.1, 0.3], [-0.35, 0.4, -0.2]]], dtype=torch.float64),
        spins=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
    )
    model = CuspGaussianOutputModel()
    terms = [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()]

    energy = local_energy(terms, model, batch)
    energy.mean().backward()

    assert energy.shape == (1,)
    assert torch.all(torch.isfinite(energy))
    assert model.alpha.grad is not None
    assert torch.isfinite(model.alpha.grad)


# --- normalize_hamiltonian_terms ---


def test_normalize_hamiltonian_terms_dict_is_used_directly() -> None:
    kinetic = KineticEnergy()
    trap = HarmonicTrap(omega=1.0)
    normalized = normalize_hamiltonian_terms({"ke": kinetic, "trap": trap})

    assert list(normalized) == ["ke", "trap"]
    assert normalized["ke"] is kinetic
    assert normalized["trap"] is trap


def test_normalize_hamiltonian_terms_list_uses_snake_case_class_names() -> None:
    normalized = normalize_hamiltonian_terms(
        [KineticEnergy(), HarmonicTrap(omega=1.0), ElectronElectronInteraction()]
    )

    assert list(normalized) == ["kinetic_energy", "harmonic_trap", "electron_electron_interaction"]


def test_normalize_hamiltonian_terms_disambiguates_repeated_classes_by_index() -> None:
    normalized = normalize_hamiltonian_terms([HarmonicTrap(omega=1.0), HarmonicTrap(omega=2.0)])

    assert list(normalized) == ["harmonic_trap_0", "harmonic_trap_1"]


def test_normalize_hamiltonian_terms_rejects_non_string_keys() -> None:
    with pytest.raises(TypeError, match="must be strings"):
        normalize_hamiltonian_terms({0: KineticEnergy()})


def test_normalize_hamiltonian_terms_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_hamiltonian_terms({"  ": KineticEnergy()})


def test_normalize_hamiltonian_terms_rejects_invalid_term_spec() -> None:
    with pytest.raises(TypeError, match="local_energy"):
        normalize_hamiltonian_terms({"bad": object()})


# --- Local-energy helper over a list of terms ---


def test_local_energy_helper_sums_terms() -> None:
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    b = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    terms = [_ConstantTerm("a", a), _ConstantTerm("b", b)]
    batch = ElectronBatch(positions=torch.zeros(3, 2, 1, dtype=torch.float64))

    total = local_energy(terms, None, batch)

    assert isinstance(total, torch.Tensor)
    assert torch.equal(total, a + b)


def test_local_energy_helper_return_terms_true_with_named_dict() -> None:
    a = torch.tensor([1.0, 2.0], dtype=torch.float64)
    b = torch.tensor([3.0, 4.0], dtype=torch.float64)
    terms = {"alpha": _ConstantTerm("alpha", a), "beta": _ConstantTerm("beta", b)}
    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    result = local_energy(terms, None, batch, return_terms=True)

    assert isinstance(result, LocalEnergyResult)
    assert torch.equal(result.total, a + b)
    # dict keys become the decomposition names
    assert torch.equal(result.terms["alpha"], a)
    assert torch.equal(result.terms["beta"], b)


def test_local_energy_helper_return_terms_true_with_list_uses_snake_class_names() -> None:
    a = torch.tensor([1.0, 2.0], dtype=torch.float64)
    b = torch.tensor([3.0, 4.0], dtype=torch.float64)
    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    # A list of two same-class terms falls back to snake-case class names,
    # disambiguated by index to keep names unique.
    result = local_energy([_ConstantTerm("a", a), _ConstantTerm("b", b)], None, batch, return_terms=True)

    assert isinstance(result, LocalEnergyResult)
    assert set(result.terms) == {"constant_term_0", "constant_term_1"}
    assert torch.equal(result.terms["constant_term_0"], a)
    assert torch.equal(result.terms["constant_term_1"], b)


def test_local_energy_rejects_empty_configured_term_name() -> None:
    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    with pytest.raises(ValueError, match="non-empty"):
        local_energy({"": _ConstantTerm("ignored", torch.zeros(2, dtype=torch.float64))}, None, batch)


def test_local_energy_rejects_term_object_without_local_energy() -> None:
    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    with pytest.raises(TypeError, match="local_energy"):
        local_energy({"bad": object()}, None, batch)


def test_local_energy_rejects_term_returning_tensor_instead_of_result() -> None:
    class TensorTerm:
        def local_energy(self, wavefunction, batch: ElectronBatch) -> torch.Tensor:
            return torch.zeros(batch.batch_size, dtype=batch.dtype)

    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    with pytest.raises(TypeError, match="LocalEnergyResult"):
        local_energy({"bad": TensorTerm()}, None, batch)


def test_local_energy_rejects_term_total_shape_mismatch() -> None:
    term = _ConstantTerm("bad", torch.zeros(2, 1, dtype=torch.float64))
    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    with pytest.raises(ValueError, match="total.*shape"):
        local_energy({"bad": term}, None, batch)


def test_local_energy_rejects_term_decomposition_shape_mismatch() -> None:
    class BadDecompositionTerm:
        def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
            total = torch.zeros(batch.batch_size, dtype=batch.dtype)
            return LocalEnergyResult(total=total, terms={"bad": torch.zeros(batch.batch_size, 1)})

    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    with pytest.raises(ValueError, match="decomposition.*shape"):
        local_energy({"bad": BadDecompositionTerm()}, None, batch)


def test_local_energy_return_terms_preserves_configured_names() -> None:
    a = torch.tensor([1.0, 2.0], dtype=torch.float64)
    b = torch.tensor([3.0, 4.0], dtype=torch.float64)
    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    result = local_energy(
        {
            "kinetic_custom": _ConstantTerm("ignored_a", a),
            "trap_custom": _ConstantTerm("ignored_b", b),
        },
        None,
        batch,
        return_terms=True,
    )

    assert isinstance(result, LocalEnergyResult)
    assert set(result.terms) == {"kinetic_custom", "trap_custom"}


# --- Term classes return decomposed LocalEnergyResult objects ---


def test_harmonic_trap_term_returns_local_energy_result() -> None:
    positions = torch.tensor(
        [[[1.0, 0.0, -1.0], [0.5, 2.0, 0.25]], [[-0.5, 1.5, 2.5], [1.0, -2.0, 0.5]]],
        dtype=torch.float64,
    )
    batch = ElectronBatch(positions=positions)
    omega = 0.75

    result = HarmonicTrap(omega=omega).local_energy(None, batch)

    expected = 0.5 * (omega**2) * positions.square().sum(dim=(1, 2))
    assert isinstance(result, LocalEnergyResult)
    assert torch.allclose(result.total, expected)
    assert torch.allclose(result.terms["harmonic_trap"], expected)


def test_electron_electron_interaction_term_returns_local_energy_result() -> None:
    positions = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    batch = ElectronBatch(positions=positions)

    result = ElectronElectronInteraction().local_energy(None, batch)

    expected = torch.linalg.norm(positions[:, 0] - positions[:, 1], dim=-1).reciprocal()
    assert isinstance(result, LocalEnergyResult)
    assert torch.allclose(result.total, expected)
    assert torch.allclose(result.terms["electron_electron"], expected)


def test_electron_nucleus_interaction_term_returns_local_energy_result() -> None:
    positions = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[-1.0, 0.5, 0.0], [0.0, 0.0, 2.0]]],
        dtype=torch.float64,
    )
    nuclei = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    charges = torch.tensor([2.0], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)

    result = ElectronNucleusInteraction(nuclei, charges).local_energy(None, batch)

    expected = -(
        charges.view(1, 1, -1)
        / torch.linalg.norm(positions.unsqueeze(2) - nuclei.view(1, 1, 1, 3), dim=-1)
    ).sum(dim=(1, 2))
    assert isinstance(result, LocalEnergyResult)
    assert torch.allclose(result.total, expected)
    assert torch.allclose(result.terms["electron_nucleus"], expected)


def test_kinetic_energy_term_returns_local_energy_result() -> None:
    positions = torch.tensor(
        [[[1.0, 2.0], [0.5, -1.0]], [[-1.5, 0.25], [2.0, 0.0]]],
        dtype=torch.float64,
    )
    batch = ElectronBatch(positions=positions)
    model = GaussianOutputModel(0.3)

    result = KineticEnergy().local_energy(model, batch)

    expected = kinetic_energy_from_logabs(model, batch)
    assert isinstance(result, LocalEnergyResult)
    assert torch.allclose(result.total, expected)
    assert torch.allclose(result.terms["kinetic"], expected)


# --- Symmetry sanity tests for exact Hooke references ---


def _two_particle_3d_batch() -> tuple[ElectronBatch, ElectronBatch]:
    """Return a batch and its particle-swapped version."""
    positions = torch.tensor(
        [
            [[0.3, -0.2, 0.8], [-0.1, 0.5, 0.3]],
            [[1.0, 0.0, 0.5], [0.0, -0.5, 1.5]],
            [[-0.4, 0.7, -0.3], [0.6, -0.1, 0.9]],
        ],
        dtype=torch.float64,
    )
    swapped = positions[:, [1, 0], :]
    return ElectronBatch(positions=positions), ElectronBatch(positions=swapped)


def test_singlet_logabs_invariant_under_particle_swap() -> None:
    wf = HookeSingletExact()
    batch, swapped = _two_particle_3d_batch()

    out = wf(batch)
    out_swapped = wf(swapped)

    assert torch.allclose(out.logabs, out_swapped.logabs)


def test_singlet_sign_invariant_under_particle_swap() -> None:
    wf = HookeSingletExact()
    batch, swapped = _two_particle_3d_batch()

    out = wf(batch)
    out_swapped = wf(swapped)

    assert torch.equal(out.sign, out_swapped.sign)


def test_triplet_logabs_invariant_under_particle_swap() -> None:
    wf = HookeTripletExact()
    # Use positions where z1 != z2 for all configs
    positions = torch.tensor(
        [
            [[0.3, -0.2, 0.8], [-0.1, 0.5, 0.3]],
            [[1.0, 0.0, 0.5], [0.0, -0.5, -0.5]],
            [[-0.4, 0.7, -0.3], [0.6, -0.1, 0.9]],
        ],
        dtype=torch.float64,
    )
    swapped = positions[:, [1, 0], :]
    batch = ElectronBatch(positions=positions)
    batch_swapped = ElectronBatch(positions=swapped)

    out = wf(batch)
    out_swapped = wf(batch_swapped)

    assert torch.allclose(out.logabs, out_swapped.logabs)


def test_triplet_sign_flips_under_particle_swap() -> None:
    wf = HookeTripletExact()
    positions = torch.tensor(
        [
            [[0.3, -0.2, 0.8], [-0.1, 0.5, 0.3]],
            [[1.0, 0.0, 0.5], [0.0, -0.5, -0.5]],
            [[-0.4, 0.7, -0.3], [0.6, -0.1, 0.9]],
        ],
        dtype=torch.float64,
    )
    swapped = positions[:, [1, 0], :]
    batch = ElectronBatch(positions=positions)
    batch_swapped = ElectronBatch(positions=swapped)

    out = wf(batch)
    out_swapped = wf(batch_swapped)

    assert torch.equal(out.sign, -out_swapped.sign)


# --- summarize_local_energy ---


def test_summarize_local_energy_all_finite() -> None:
    eloc = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)

    metrics = summarize_local_energy(eloc)

    assert metrics["n_samples"] == 3
    assert metrics["n_finite_samples"] == 3
    assert metrics["nonfinite_energy_fraction"] == 0.0
    assert metrics["energy_mean"] == pytest.approx(3.0)
    assert metrics["energy_variance"] == pytest.approx(8.0 / 3.0)
    assert metrics["energy_stderr"] == pytest.approx(math.sqrt(8.0 / 3.0) / math.sqrt(3))
    assert "expected_energy" not in metrics


def test_summarize_local_energy_some_nonfinite() -> None:
    eloc = torch.tensor([1.0, float("inf"), 3.0, float("nan")], dtype=torch.float64)

    metrics = summarize_local_energy(eloc)

    assert metrics["n_samples"] == 4
    assert metrics["n_finite_samples"] == 2
    assert metrics["nonfinite_energy_fraction"] == pytest.approx(0.5)
    assert metrics["energy_mean"] == pytest.approx(2.0)
    assert math.isfinite(metrics["energy_variance"])


def test_summarize_local_energy_all_nonfinite() -> None:
    eloc = torch.tensor([float("inf"), float("nan")], dtype=torch.float64)

    metrics = summarize_local_energy(eloc)

    assert metrics["n_finite_samples"] == 0
    assert metrics["nonfinite_energy_fraction"] == 1.0
    assert math.isnan(metrics["energy_mean"])
    assert math.isnan(metrics["energy_variance"])
    assert metrics["energy_stderr"] == float("inf")


def test_summarize_local_energy_empty() -> None:
    metrics = summarize_local_energy(torch.empty(0, dtype=torch.float64))

    assert metrics["n_samples"] == 0
    assert metrics["n_finite_samples"] == 0
    assert math.isnan(metrics["energy_mean"])
    assert math.isnan(metrics["energy_variance"])
    assert metrics["energy_stderr"] == float("inf")
    assert math.isnan(metrics["nonfinite_energy_fraction"])


def test_summarize_local_energy_is_reference_free() -> None:
    eloc = torch.tensor([1.5, 2.5], dtype=torch.float64)

    metrics = summarize_local_energy(eloc)

    assert "reference_energy" not in metrics
    assert "energy_error" not in metrics
    assert "abs_energy_error" not in metrics


def test_reference_energy_metrics_computes_signed_and_absolute_error() -> None:
    metrics = reference_energy_metrics(energy_mean=2.5, reference_energy=2.0)

    assert metrics["reference_energy"] == 2.0
    assert metrics["energy_error"] == pytest.approx(0.5)
    assert metrics["abs_energy_error"] == pytest.approx(0.5)


def test_reference_energy_metrics_absolute_error_is_nonnegative() -> None:
    metrics = reference_energy_metrics(energy_mean=1.5, reference_energy=2.0)

    assert metrics["energy_error"] == pytest.approx(-0.5)
    assert metrics["abs_energy_error"] == pytest.approx(0.5)


def test_summarize_local_energy_result_with_finite_terms() -> None:
    result = LocalEnergyResult(
        total=torch.tensor([2.0, 2.0], dtype=torch.float64),
        terms={
            "kinetic": torch.tensor([1.0, 1.0], dtype=torch.float64),
            "trap": torch.tensor([1.0, 1.0], dtype=torch.float64),
        },
    )

    metrics = summarize_local_energy(result)

    assert metrics["terms.kinetic_mean"] == pytest.approx(1.0)
    assert metrics["terms.kinetic_nonfinite_fraction"] == 0.0
    assert metrics["terms.trap_mean"] == pytest.approx(1.0)
    assert metrics["terms.trap_nonfinite_fraction"] == 0.0


def test_summarize_local_energy_result_with_nonfinite_term_values() -> None:
    result = LocalEnergyResult(
        total=torch.tensor([2.0, 2.0], dtype=torch.float64),
        terms={"kinetic": torch.tensor([1.0, float("nan")], dtype=torch.float64)},
    )

    metrics = summarize_local_energy(result)

    assert metrics["terms.kinetic_mean"] == pytest.approx(1.0)
    assert metrics["terms.kinetic_nonfinite_fraction"] == pytest.approx(0.5)


def test_summarize_local_energy_term_with_no_finite_values() -> None:
    result = LocalEnergyResult(
        total=torch.tensor([2.0], dtype=torch.float64),
        terms={"kinetic": torch.tensor([float("nan")], dtype=torch.float64)},
    )

    metrics = summarize_local_energy(result)

    assert math.isnan(metrics["terms.kinetic_mean"])
    assert metrics["terms.kinetic_nonfinite_fraction"] == 1.0
