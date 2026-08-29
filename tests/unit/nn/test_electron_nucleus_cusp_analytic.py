"""Typed analytic electron-nucleus cusp capability tests."""

import pytest
import torch

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch
from tpen.data.permutation import Permutation
from tpen.nn.cusp import (
    CurvatureElectronNucleusCuspLaw,
    ElectronNucleusCusp,
    ElectronNucleusCuspEvaluation,
    LinearElectronNucleusCuspLaw,
)


def _atoms() -> AtomicConfiguration:
    return AtomicConfiguration(
        positions=torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float64),
        charges=torch.tensor([2.0, 1.0], dtype=torch.float64),
    )


def _batch() -> ElectronBatch:
    return ElectronBatch(
        positions=torch.tensor([[[0.5, 0.0], [1.0, 1.0]]], dtype=torch.float64)
    )


def test_linear_analytic_evaluation_has_exact_terms_and_origin_slope() -> None:
    evaluation = ElectronNucleusCusp(_atoms(), LinearElectronNucleusCuspLaw()).analytic_evaluation(_batch())

    assert evaluation.displacement.shape == (1, 2, 2, 2)
    assert evaluation.distance.shape == (1, 2, 2)
    torch.testing.assert_close(evaluation.radial_first_derivative, -evaluation.nuclear_charges.view(1, 1, 2))
    assert torch.equal(evaluation.radial_second_derivative, torch.zeros_like(evaluation.distance))
    assert torch.equal(evaluation.slope_residual, torch.zeros_like(evaluation.distance))
    torch.testing.assert_close(evaluation.origin_radial_slope, -evaluation.nuclear_charges)
    torch.testing.assert_close(evaluation.local_energy_pair(), torch.zeros_like(evaluation.distance))


def test_curvature_analytic_evaluation_uses_cancelled_residual_at_origin() -> None:
    law = CurvatureElectronNucleusCuspLaw(curvature_coefficient=0.3, curvature_range=1.5)
    evaluation = ElectronNucleusCusp(_atoms(), law).analytic_evaluation(_batch())
    r = evaluation.distance
    c, d = law.curvature_coefficient, law.curvature_range
    denominator = 1.0 + d * r

    torch.testing.assert_close(evaluation.pair_value, law.value(r, _atoms().charges.view(1, 1, 2)))
    torch.testing.assert_close(evaluation.radial_first_derivative, -_atoms().charges.view(1, 1, 2) + c * r * (2 + d * r) / denominator.square())
    torch.testing.assert_close(evaluation.radial_second_derivative, 2 * c / denominator.pow(3))
    torch.testing.assert_close(evaluation.slope_residual, c * (2 + d * r) / denominator.square())

    origin = ElectronNucleusCuspEvaluation(
        displacement=torch.zeros(1, 1, 1, 2, dtype=torch.float64),
        distance=torch.zeros(1, 1, 1, dtype=torch.float64),
        pair_value=torch.zeros(1, 1, 1, dtype=torch.float64),
        radial_first_derivative=torch.zeros(1, 1, 1, dtype=torch.float64),
        radial_second_derivative=2 * c.expand(1, 1, 1),
        slope_residual=(2 * c).expand(1, 1, 1),
        nuclear_charges=torch.ones(1, dtype=torch.float64),
        origin_radial_slope=-torch.ones(1, dtype=torch.float64),
    )
    torch.testing.assert_close(origin.local_energy_pair(), torch.full((1, 1, 1), -3 * c))


def test_curvature_analytic_request_reaches_both_trainable_parameters() -> None:
    law = CurvatureElectronNucleusCuspLaw(curvature_coefficient=0.3, curvature_range=1.5)
    evaluation = ElectronNucleusCusp(_atoms(), law).analytic_evaluation(_batch())
    evaluation.pair_value.sum().backward()

    assert law.raw_curvature_coefficient.grad is not None
    assert law.raw_curvature_range.grad is not None
    assert torch.isfinite(law.raw_curvature_coefficient.grad)
    assert torch.isfinite(law.raw_curvature_range.grad)


def test_evaluation_permute_and_compare_cover_all_electron_fields() -> None:
    evaluation = ElectronNucleusCusp(_atoms()).analytic_evaluation(_batch())
    permuted = evaluation.permute(Permutation((1, 0)))
    round_trip = permuted.permute(Permutation((1, 0)))

    close, metrics = evaluation.compare(round_trip)
    assert close
    assert metrics["max_abs_error"] == pytest.approx(0.0)
    assert permuted.n_electrons == evaluation.n_electrons
    with pytest.raises(ValueError, match="incompatible"):
        evaluation.permute(Permutation((0, 1, 2)))


def test_analytic_factor_output_is_invariant_to_nucleus_relabeling() -> None:
    atoms = _atoms()
    relabeled = AtomicConfiguration(positions=atoms.positions.flip(0), charges=atoms.charges.flip(0))
    batch = _batch()
    left = ElectronNucleusCusp(atoms, LinearElectronNucleusCuspLaw()).analytic_evaluation(batch)
    right = ElectronNucleusCusp(relabeled, LinearElectronNucleusCuspLaw()).analytic_evaluation(batch)

    torch.testing.assert_close(left.pair_value.sum(dim=(1, 2)), right.pair_value.sum(dim=(1, 2)))
    torch.testing.assert_close(left.local_energy_pair().sum(dim=(1, 2)), right.local_energy_pair().sum(dim=(1, 2)))


def test_ordinary_forward_does_not_request_analytic_capability(monkeypatch) -> None:
    cusp = ElectronNucleusCusp(_atoms())
    calls = 0
    original = cusp.analytic_evaluation

    def spy(batch):
        nonlocal calls
        calls += 1
        return original(batch)

    monkeypatch.setattr(cusp, "analytic_evaluation", spy)
    cusp(_batch())
    assert calls == 0
    cusp.analytic_evaluation(_batch())
    assert calls == 1


def test_origin_slope_is_independent_of_mutated_value_law(monkeypatch) -> None:
    law = CurvatureElectronNucleusCuspLaw(curvature_coefficient=0.3, curvature_range=1.5)
    charges = torch.tensor([2.0], dtype=torch.float64)
    zero = torch.zeros(1, 1, 1, dtype=torch.float64)
    small = torch.full((1, 1, 1), 1.0e-8, dtype=torch.float64)
    origin = law.origin_radial_slope(charges)
    first_at_zero = law.analytic_terms(zero, charges.view(1, 1, 1))[1]
    first_at_small = law.analytic_terms(small, charges.view(1, 1, 1))[1]

    torch.testing.assert_close(first_at_zero, origin.view(1, 1, 1), atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(first_at_small, origin.view(1, 1, 1), atol=1.0e-7, rtol=0.0)

    # `analytic_terms` owns the derivative provider, and its closed form does
    # not differentiate `value`. Mutating only `value` therefore cannot move
    # this derivative; mutate the provider's slope to model a value-law edit
    # whose analytic derivative was left stale.
    original_terms = law.analytic_terms

    def mutated_terms(distance, supplied_charges):
        value, first, second, residual = original_terms(distance, supplied_charges)
        return 2.0 * value, 2.0 * first, second, residual

    monkeypatch.setattr(law, "analytic_terms", mutated_terms)
    mutated_first = law.analytic_terms(zero, charges.view(1, 1, 1))[1]
    assert not torch.allclose(mutated_first, origin.view(1, 1, 1), atol=1.0e-7, rtol=0.0)


def test_curvature_residual_is_pre_cancelled_at_float64_small_radii() -> None:
    law = CurvatureElectronNucleusCuspLaw(curvature_coefficient=0.3, curvature_range=1.5)
    charges = torch.tensor([2.0], dtype=torch.float64)
    radii = torch.tensor([1.0e-6, 1.0e-8, 1.0e-10], dtype=torch.float64).view(1, 1, 3)
    broadcast_charges = charges.view(1, 1, 1)
    _, first, _, residual = law.analytic_terms(radii, broadcast_charges)
    c, d = law.curvature_coefficient, law.curvature_range
    expected = c * (2.0 + d * radii) / (1.0 + d * radii).square()
    naive = (first + broadcast_charges) / radii
    cancelled_error = (residual - expected).abs()
    naive_error = (naive - expected).abs()

    torch.testing.assert_close(residual, expected, atol=1.0e-14, rtol=1.0e-14)
    # The smallest radius is deliberately below the scale where subtracting
    # the nearly equal slopes is harmless. This guards against regressing to
    # `(first + Z) / r`, while the first assertion guards the provider form.
    assert naive_error[..., -1] > 1.0e-7
    assert naive_error[..., -1] > 10.0 * torch.clamp(cancelled_error[..., -1], min=1.0e-16)
