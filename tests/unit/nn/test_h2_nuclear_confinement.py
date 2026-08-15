"""H2-only bounded-cusp and smooth-tail confinement tests."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from typeguard import TypeCheckError

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.permutation import Permutation, all_permutations
from tpen.data.real import Feature
from tpen.nn import (
    AdditiveEnvelope,
    Embedding,
    H2NuclearConfinement,
    H2NuclearConfinementEvaluation,
    H2NuclearFactorizedEnvelope,
    H2NuclearFactorizedWavefunctionParts,
    TPENWaveFunction,
)
from tpen.nn.readout import PfaffianReadout
from tests.helpers.equivariance import assert_equivariant_all


class EmptyEncoder(nn.Module):
    def forward(self, batch: ElectronBatch, *, context=None) -> Feature:
        return Feature()


class CountingReadout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, features: Feature, batch: ElectronBatch) -> WavefunctionOutput:
        self.calls += 1
        sign = torch.sign(batch.positions[:, 0, 0] - batch.positions[:, 1, 0])
        return WavefunctionOutput(logabs=torch.zeros_like(sign), sign=sign, aux={"calls": self.calls})


def _factor() -> H2NuclearConfinement:
    return H2NuclearConfinement(beta_H=1.7, a=0.8, kappa=0.6)


def _batch(*, sampled_geometry: bool = False) -> ElectronBatch:
    positions = torch.tensor(
        [
            [[0.2, -0.3, 0.4], [1.1, 0.2, -0.1], [-0.4, 0.5, 0.3]],
            [[0.7, 0.1, -0.6], [1.8, -0.2, 0.5], [0.1, 0.8, -0.2]],
        ],
        dtype=torch.float64,
    )
    nuclei = torch.tensor(
        [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64
    )
    charges = torch.ones(2, dtype=torch.float64)
    if sampled_geometry:
        nuclei = torch.stack([nuclei, nuclei + nuclei.new_tensor([0.1, -0.2, 0.3])])
        charges = charges.expand(2, -1).clone()
    spins = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], dtype=torch.float64)
    return ElectronBatch(
        positions=positions,
        nuclear_positions=nuclei,
        nuclear_charges=charges,
        spins=spins,
    )


@pytest.mark.parametrize("name", ["beta_H", "a", "kappa"])
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_h2_parameters_are_explicit_finite_positive_shared_scalars(name: str, bad: float) -> None:
    values = {"beta_H": 1.0, "a": 1.0, "kappa": 1.0}
    values[name] = bad

    with pytest.raises(ValueError, match=name):
        H2NuclearConfinement(**values)


def test_h2_parameters_reject_index_specific_values_and_are_checkpointed() -> None:
    with pytest.raises((TypeError, TypeCheckError)):
        H2NuclearConfinement(beta_H=[1.0, 1.0], a=1.0, kappa=1.0)  # type: ignore[arg-type]

    factor = _factor()
    assert tuple(factor.state_dict()) == ("beta_H", "a", "kappa")
    assert factor.beta_H.ndim == factor.a.ndim == factor.kappa.ndim == 0


def test_h2_geometry_requires_two_distinct_unit_charge_centres() -> None:
    positions = torch.zeros(1, 2, 3, dtype=torch.float64)
    factor = _factor()

    with pytest.raises(ValueError, match="nuclear_positions"):
        factor(ElectronBatch(positions=positions))
    with pytest.raises(ValueError, match="nuclear_charges"):
        factor(
            ElectronBatch(
                positions=positions,
                nuclear_positions=torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]),
            )
        )
    with pytest.raises(ValueError, match="exactly two"):
        factor(
            ElectronBatch(
                positions=positions,
                nuclear_positions=torch.zeros(1, 3),
                nuclear_charges=torch.ones(1),
            )
        )
    with pytest.raises(ValueError, match="exactly two"):
        factor(
            ElectronBatch(
                positions=positions,
                nuclear_positions=torch.zeros(3, 3),
                nuclear_charges=torch.ones(3),
            )
        )
    with pytest.raises(ValueError, match="unit H charges"):
        factor(
            ElectronBatch(
                positions=positions,
                nuclear_positions=torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]),
                nuclear_charges=torch.tensor([1.0, 2.0]),
            )
        )
    with pytest.raises(ValueError, match="nondegenerate"):
        factor(
            ElectronBatch(
                positions=positions,
                nuclear_positions=torch.zeros(2, 3),
                nuclear_charges=torch.ones(2),
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("nuclear_positions", [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]], "finite nuclear"),
        ("nuclear_charges", [1.0, float("inf")], "finite nuclear charges"),
    ],
)
def test_h2_geometry_rejects_nonfinite_metadata(field: str, value, message: str) -> None:
    nuclei = torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64)
    charges = torch.ones(2, dtype=torch.float64)
    if field == "nuclear_positions":
        nuclei = torch.tensor(value, dtype=torch.float64)
    else:
        charges = torch.tensor(value, dtype=torch.float64)
    batch = ElectronBatch(
        positions=torch.zeros(1, 2, 3, dtype=torch.float64),
        nuclear_positions=nuclei,
        nuclear_charges=charges,
    )

    with pytest.raises(ValueError, match=message):
        _factor()(batch)


@pytest.mark.parametrize("sampled_geometry", [False, True])
def test_vectorized_h2_evaluation_matches_slow_reference(sampled_geometry: bool) -> None:
    batch = _batch(sampled_geometry=sampled_geometry)
    factor = _factor()

    fast = factor.evaluate(batch)
    slow = factor.evaluate_reference(batch)

    assert isinstance(fast, H2NuclearConfinementEvaluation)
    close, metrics = fast.compare(slow, atol=2.0e-15, rtol=2.0e-15)
    assert close, metrics
    torch.testing.assert_close(
        factor(batch),
        slow.bounded_cusp_logabs() + slow.smooth_tail_logabs.sum(dim=1),
        atol=2.0e-15,
        rtol=2.0e-15,
    )


def test_raw_exact_and_near_zero_factor_data_are_finite_with_explicit_domain_split() -> None:
    nuclei = torch.tensor([[0.0, 0.0], [1.4, 0.0]], dtype=torch.float64)
    exact = ElectronBatch(
        positions=torch.tensor([[[0.0, 0.0], [1.4, 0.0]]], dtype=torch.float64),
        nuclear_positions=nuclei,
        nuclear_charges=torch.ones(2, dtype=torch.float64),
    )
    factor = _factor()

    evaluation = factor.evaluate(exact)
    reference = factor.evaluate_reference(exact)

    close, metrics = evaluation.compare(reference, atol=0.0, rtol=0.0)
    assert close, metrics
    assert evaluation.distance[0, 0, 0].item() == 0.0
    assert evaluation.distance[0, 1, 1].item() == 0.0
    assert evaluation.value[0, 0, 0].item() == 0.0
    assert evaluation.value[0, 1, 1].item() == 0.0
    torch.testing.assert_close(
        evaluation.radial_first_derivative[0, (0, 1), (0, 1)],
        torch.full((2,), -1.0, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        evaluation.radial_second_derivative[0, (0, 1), (0, 1)],
        torch.full((2,), 1.7, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        evaluation.cusp_residual[0, (0, 1), (0, 1)],
        torch.full((2,), 1.7, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.isfinite(factor(exact)).all()
    with pytest.raises(ValueError, match="exact electron-nucleus coalescence"):
        evaluation.require_separated_local_energy_domain()

    near = ElectronBatch(
        positions=torch.tensor([[[1.0e-14, 0.0], [1.4 - 1.0e-14, 0.0]]], dtype=torch.float64),
        nuclear_positions=nuclei,
        nuclear_charges=torch.ones(2, dtype=torch.float64),
    )
    near_evaluation = factor.evaluate(near)
    near_reference = factor.evaluate_reference(near)
    close, metrics = near_evaluation.compare(near_reference, atol=2.0e-15, rtol=2.0e-15)
    assert close, metrics
    assert near_evaluation.distance[0, 0, 0] < 1.0e-12
    assert near_evaluation.distance[0, 0, 0].item() > 0.0
    assert near_evaluation.require_separated_local_energy_domain() is near_evaluation
    torch.testing.assert_close(
        near_evaluation.cusp_residual[0, 0, 0],
        torch.tensor(1.7, dtype=torch.float64),
        atol=2.0e-14,
        rtol=0.0,
    )


def test_nonlinear_radial_derivatives_match_float64_autodiff_at_each_centre() -> None:
    positions = torch.tensor(
        [[[0.37, 0.0, 0.0], [1.77, 0.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    batch = ElectronBatch(
        positions=positions,
        nuclear_positions=torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float64),
        nuclear_charges=torch.ones(2, dtype=torch.float64),
    )
    evaluation = _factor().evaluate(batch)

    for electron_index, nucleus_index in ((0, 0), (1, 1)):
        cusp = evaluation.value[0, electron_index, nucleus_index]
        first = torch.autograd.grad(cusp, positions, create_graph=True, retain_graph=True)[0][
            0, electron_index, 0
        ]
        second = torch.autograd.grad(first, positions, retain_graph=True)[0][0, electron_index, 0]
        torch.testing.assert_close(
            first,
            evaluation.radial_first_derivative[0, electron_index, nucleus_index],
            atol=2.0e-15,
            rtol=2.0e-15,
        )
        torch.testing.assert_close(
            second,
            evaluation.radial_second_derivative[0, electron_index, nucleus_index],
            atol=2.0e-15,
            rtol=2.0e-15,
        )
        assert evaluation.radial_second_derivative[0, electron_index, nucleus_index] > 0


def test_complete_h2_factor_has_the_kato_spherical_slope_at_each_centre() -> None:
    factor = _factor()
    nuclei = torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64)
    directions = torch.cat([torch.eye(3, dtype=torch.float64), -torch.eye(3, dtype=torch.float64)])
    epsilon = 1.0e-6

    for nucleus_index in range(2):
        centre = nuclei[nucleus_index]
        origin = ElectronBatch(
            positions=centre.reshape(1, 1, 3),
            nuclear_positions=nuclei,
            nuclear_charges=torch.ones(2, dtype=torch.float64),
        )
        shells = ElectronBatch(
            positions=(centre + epsilon * directions).reshape(6, 1, 3),
            nuclear_positions=nuclei,
            nuclear_charges=torch.ones(2, dtype=torch.float64),
        )
        spherical_slope = ((factor(shells) - factor(origin)) / epsilon).mean()
        torch.testing.assert_close(
            spherical_slope,
            torch.tensor(-1.0, dtype=torch.float64),
            atol=3.0e-6,
            rtol=0.0,
        )


def test_smooth_tail_is_quadratic_at_centres_and_has_kappa_far_slope_in_directions() -> None:
    factor = H2NuclearConfinement(beta_H=1.3, a=0.9, kappa=0.75)
    nuclei = torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64)
    exact = ElectronBatch(
        positions=nuclei.reshape(1, 2, 3),
        nuclear_positions=nuclei,
        nuclear_charges=torch.ones(2, dtype=torch.float64),
    )
    tail = factor.evaluate(exact).smooth_tail_logabs
    torch.testing.assert_close(tail, torch.zeros_like(tail), atol=0.0, rtol=0.0)

    epsilon = 1.0e-5
    near = ElectronBatch(
        positions=torch.tensor([[[-0.7 + epsilon, 0.0, 0.0], [0.7 - epsilon, 0.0, 0.0]]], dtype=torch.float64),
        nuclear_positions=nuclei,
        nuclear_charges=torch.ones(2, dtype=torch.float64),
    )
    near_tail = factor.evaluate(near).smooth_tail_logabs
    assert torch.all(near_tail.abs() < 2.0 * epsilon**2)

    for direction in (
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
    ):
        radii = torch.tensor([10_000.0, 10_001.0], dtype=torch.float64)
        positions = (radii[:, None] * direction).reshape(2, 1, 3)
        batch = ElectronBatch(
            positions=positions,
            nuclear_positions=nuclei,
            nuclear_charges=torch.ones(2, dtype=torch.float64),
        )
        values = factor(batch)
        slope = values[1] - values[0]
        torch.testing.assert_close(
            slope,
            torch.tensor(-0.75, dtype=torch.float64),
            atol=1.0e-8,
            rtol=0.0,
        )


def test_electron_and_identical_h_permutations_use_explicit_typed_contracts() -> None:
    batch = _batch()
    factor = _factor()
    electron_permutation = Permutation((2, 0, 1))
    nucleus_permutation = Permutation((1, 0))
    evaluation = factor.evaluate(batch)

    electron_lhs = factor.evaluate(batch.permute(electron_permutation))
    electron_rhs = evaluation.permute(electron_permutation)
    close, metrics = electron_lhs.compare(electron_rhs, atol=0.0, rtol=0.0)
    assert close, metrics

    assert batch.nuclear_positions is not None and batch.nuclear_charges is not None
    swapped_batch = ElectronBatch(
        positions=batch.positions,
        nuclear_positions=batch.nuclear_positions[[1, 0]],
        nuclear_charges=batch.nuclear_charges[[1, 0]],
        spins=batch.spins,
    )
    swapped = factor.evaluate(swapped_batch)
    relabelled = evaluation.permute_nuclei(nucleus_permutation)
    close, metrics = swapped.compare(relabelled, atol=0.0, rtol=0.0)
    assert close, metrics
    torch.testing.assert_close(factor(batch), factor(swapped_batch), atol=0.0, rtol=0.0)


def test_h2_factorization_keeps_tail_regular_and_reads_out_once_without_aliasing() -> None:
    batch = _batch()
    readout = CountingReadout()
    factor = _factor()
    model = TPENWaveFunction(
        embedding=EmptyEncoder(),
        layers=[nn.Identity()],
        readout=readout,
        h2_nuclear_envelope=H2NuclearFactorizedEnvelope(AdditiveEnvelope(), factor),
    )

    parts = model.h2_nuclear_factorization(batch)
    assert isinstance(parts, H2NuclearFactorizedWavefunctionParts)
    assert readout.calls == 1
    torch.testing.assert_close(
        parts.regular_logabs,
        parts.nuclear.smooth_tail_logabs.sum(dim=1),
        atol=0.0,
        rtol=0.0,
    )
    output = parts.as_output()
    torch.testing.assert_close(
        output.logabs,
        parts.regular_logabs + parts.nuclear.bounded_cusp_logabs(),
        atol=0.0,
        rtol=0.0,
    )
    output.aux["mutation"] = True
    assert "mutation" not in parts.aux

    ordinary = model(batch)
    assert readout.calls == 2
    torch.testing.assert_close(ordinary.logabs, parts.as_output().logabs, atol=2.0e-15, rtol=2.0e-15)


def test_ordinary_h2_factor_and_model_forward_do_not_invoke_derivative_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _batch()
    factor = _factor()
    model = TPENWaveFunction(
        embedding=EmptyEncoder(),
        layers=[nn.Identity()],
        readout=CountingReadout(),
        h2_nuclear_envelope=H2NuclearFactorizedEnvelope(AdditiveEnvelope(), factor),
    )

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("ordinary H2 forward must not evaluate radial derivatives")

    monkeypatch.setattr(factor, "evaluate", _forbidden)
    assert torch.isfinite(factor(batch)).all()
    assert torch.isfinite(model(batch).logabs).all()
    with pytest.raises(AssertionError, match="radial derivatives"):
        model.h2_nuclear_factorization(batch)


def test_real_h2_tpen_model_is_fully_fermionic_and_identical_h_invariant() -> None:
    torch.manual_seed(19)
    factor = _factor()
    model = TPENWaveFunction(
        embedding=Embedding(
            max_order=2,
            spatial_dim=3,
            out_channels=2,
            hidden_channels=5,
            num_hidden_layers=1,
            include_spins=True,
        ),
        layers=(),
        readout=PfaffianReadout(channels=2),
        h2_nuclear_envelope=H2NuclearFactorizedEnvelope(AdditiveEnvelope(), factor),
    ).to(dtype=torch.float64)
    batch = ElectronBatch(
        positions=torch.tensor(
            [[[0.2, -0.3, 0.4], [1.1, 0.2, -0.1]], [[-0.4, 0.5, 0.3], [0.8, -0.7, 0.6]]],
            dtype=torch.float64,
        ),
        nuclear_positions=torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64),
        nuclear_charges=torch.ones(2, dtype=torch.float64),
        spins=torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float64),
    )
    output = model(batch)

    assert_equivariant_all(model, batch, atol=1.0e-10, rtol=1.0e-10)
    for permutation in all_permutations(batch.n_electrons):
        permuted = model(batch.permute(permutation))
        torch.testing.assert_close(permuted.logabs, output.logabs, atol=1.0e-10, rtol=1.0e-10)
        torch.testing.assert_close(permuted.sign, output.sign * permutation.sign)

    assert batch.nuclear_positions is not None and batch.nuclear_charges is not None
    swapped_batch = ElectronBatch(
        positions=batch.positions,
        nuclear_positions=batch.nuclear_positions[[1, 0]],
        nuclear_charges=batch.nuclear_charges[[1, 0]],
        spins=batch.spins,
    )
    swapped = model(swapped_batch)
    torch.testing.assert_close(swapped.logabs, output.logabs, atol=0.0, rtol=0.0)
    torch.testing.assert_close(swapped.sign, output.sign, atol=0.0, rtol=0.0)

    parts = model.h2_nuclear_factorization(batch)
    permuted_parts = model.h2_nuclear_factorization(batch.permute(Permutation((1, 0))))
    expected_parts = parts.permute(Permutation((1, 0)))
    close, metrics = permuted_parts.compare(expected_parts, atol=1.0e-10, rtol=1.0e-10)
    assert close, metrics

    swapped_parts = model.h2_nuclear_factorization(swapped_batch)
    expected_swapped_parts = parts.permute_nuclei(Permutation((1, 0)))
    close, metrics = swapped_parts.compare(expected_swapped_parts, atol=0.0, rtol=0.0)
    assert close, metrics


def test_h2_parameter_formula_matches_approved_closed_form() -> None:
    factor = _factor()
    evaluation = factor.evaluate(_batch())
    distance = evaluation.distance
    expected_value = torch.expm1(-1.7 * distance) / 1.7
    expected_first = -torch.exp(-1.7 * distance)
    expected_second = 1.7 * torch.exp(-1.7 * distance)
    expected_residual = -torch.expm1(-1.7 * distance) / distance

    torch.testing.assert_close(evaluation.value, expected_value)
    torch.testing.assert_close(evaluation.radial_first_derivative, expected_first)
    torch.testing.assert_close(evaluation.radial_second_derivative, expected_second)
    torch.testing.assert_close(evaluation.cusp_residual, expected_residual)
    assert math.isclose(float(factor.kappa), 0.6)
