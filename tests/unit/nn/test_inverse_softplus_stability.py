"""Stability of the shared inverse-softplus parameter initializer.

The defect this pins was NOT that ``log(expm1(x))`` overflows -- it is that the
overflow was SILENT. ``expm1(x)`` returns ``inf`` above roughly 709.78 in
float64, so a constructor stored a non-finite raw parameter, construction
succeeded, and every downstream value and gradient became NaN. A run then
proceeded and poisoned its numbers instead of failing where the bad input
entered.

The boundary here is the one an independent verifier MEASURED (709 finite, 710
NaN), so these tests bind to observed behaviour rather than to the argument
that produced the fix.
"""

from __future__ import annotations

import math

import pytest
import torch

from tpen.nn.cusp import ElectronElectronCusp, TrainableCurvatureElectronNucleusCuspLaw
from tpen.nn.envelope import GaussianConfinement
from tpen.nn.factor import _INVERSE_SOFTPLUS_LARGE_INPUT, _inverse_softplus


def _legacy_inverse_softplus(value: float) -> torch.Tensor:
    """The pre-fix implementation, kept as the small-x reference.

    A5's bit-identity gate and the additive checkpoint-compatibility guarantee
    both depend on ordinary O(1) initializations being unchanged, so the small-x
    path is compared against this rather than against a fresh derivation.
    """

    value = max(value, 1e-12)
    return torch.log(torch.expm1(torch.tensor(value, dtype=torch.float64)))


@pytest.mark.parametrize("value", [709.0, 710.0, 1000.0, 1e4, 1e8, 1e300])
def test_large_inputs_are_finite(value: float) -> None:
    """Every representable positive input yields a finite raw parameter.

    710 is the measured first NaN of the previous implementation; 1e300 is far
    past any plausible configuration and still has an exact answer.
    """

    raw = _inverse_softplus(value)
    assert torch.isfinite(raw), f"raw parameter for {value} is not finite: {raw}"


@pytest.mark.parametrize("value", [709.0, 710.0, 1000.0, 1e4])
def test_large_inputs_round_trip_through_softplus(value: float) -> None:
    """The raw parameter's softplus returns the requested value.

    Finiteness alone would be satisfied by any arbitrary finite number; this is
    what makes the returned value CORRECT rather than merely non-NaN.
    """

    recovered = torch.nn.functional.softplus(_inverse_softplus(value))
    assert recovered.item() == pytest.approx(value, rel=1e-12)


def test_the_previous_implementation_really_did_produce_nan_at_710() -> None:
    """Guard the premise: 709 finite and 710 NaN under the OLD computation.

    Without this, the boundary tests above could pass against an implementation
    that never had a boundary, and the suite would be asserting nothing.
    """

    assert torch.isfinite(_legacy_inverse_softplus(709.0))
    assert not torch.isfinite(_legacy_inverse_softplus(710.0))


@pytest.mark.parametrize(
    "value",
    [1e-12, 1e-6, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 19.0, 20.0],
)
def test_small_inputs_are_bitwise_unchanged(value: float) -> None:
    """Ordinary initializations must be BITWISE identical to the old path.

    A5's bit-identity receipt and the checkpoint-compatibility guarantee both
    rest on this: a change here would be a regression of this slice, not an
    improvement to it.
    """

    assert _inverse_softplus(value).item() == _legacy_inverse_softplus(value).item()


def test_below_floor_inputs_are_bitwise_unchanged() -> None:
    """The 1e-12 floor keeps its previous behaviour for non-positive input."""

    for value in (0.0, -1.0, 1e-30):
        assert _inverse_softplus(value).item() == _legacy_inverse_softplus(value).item()


def test_branch_is_continuous_at_the_crossover() -> None:
    """The two forms agree to sub-ulp accuracy where the branch switches.

    A discontinuity at the crossover would mean two configurations either side
    of an implementation detail initialize measurably differently.
    """

    crossover = _INVERSE_SOFTPLUS_LARGE_INPUT
    below = _inverse_softplus(crossover).item()
    just_above = _inverse_softplus(math.nextafter(crossover, math.inf)).item()
    assert just_above == pytest.approx(below, rel=0.0, abs=1e-15)


@pytest.mark.parametrize("range_value", [710.0, 1e4])
def test_en_cusp_trainable_range_is_finite_at_large_init(range_value: float) -> None:
    """Call site 1: `TrainableCurvatureElectronNucleusCuspLaw.raw_curvature_range`."""

    law = TrainableCurvatureElectronNucleusCuspLaw(
        curvature_coefficient=1e-3, curvature_range=range_value, trainable=True
    )
    assert torch.isfinite(law.raw_curvature_range).all()
    assert torch.isfinite(law.curvature_range).all()


@pytest.mark.parametrize("range_value", [710.0, 1e4])
def test_ee_cusp_both_ranges_are_finite_at_large_init(range_value: float) -> None:
    """Call sites 2 and 3: `raw_same_range` and `raw_opposite_range`.

    Both are exercised because both sit on the ``trainable_range: true`` path
    the production arm may enable; testing one would leave the other unpinned.
    """

    cusp = ElectronElectronCusp(range_parameter=range_value, trainable_range=True)
    assert torch.isfinite(cusp.raw_same_range).all()
    assert torch.isfinite(cusp.raw_opposite_range).all()


def test_ee_cusp_asymmetric_ranges_are_each_finite() -> None:
    """The two ee ranges are initialized independently, so vary them apart."""

    cusp = ElectronElectronCusp(
        same_range_parameter=710.0,
        opposite_range_parameter=5000.0,
        trainable_range=True,
    )
    assert torch.isfinite(cusp.raw_same_range).all()
    assert torch.isfinite(cusp.raw_opposite_range).all()


@pytest.mark.parametrize("coefficient", [710.0, 1e4])
def test_envelope_coefficient_is_finite_at_large_init(coefficient: float) -> None:
    """Call site 4: `GaussianConfinement.raw_coefficient`."""

    envelope = GaussianConfinement(coefficient=coefficient, trainable=True)
    assert torch.isfinite(envelope.raw_coefficient).all()
