"""Contracts for the trainable-factor scalar trace.

The measurement this enables: convergence adequacy assessed on the loss alone
cannot see a parameter the loss is insensitive to. A cusp range well away from
its optimum can cost far less energy than the window-to-window scatter of a
training trace, so the loss plateaus on schedule while the model is still
structurally in transit -- and the primary checkpoint cannot be moved after the
fact. Same reason initialization bias is tested on the sampled MEAN rather than
on tau or split-Rhat: the instrument has to be able to see the thing.
"""

from __future__ import annotations

import torch

from tpen.callback.factor_scalars import collect_factor_scalars
from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.nn import (
    ElectronElectronCusp,
    ElectronNucleusCusp,
    LinearElectronNucleusCuspLaw,
    CurvatureElectronNucleusCuspLaw,
)


def _atoms() -> AtomicConfiguration:
    return AtomicConfiguration(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        charges=torch.tensor([2.0], dtype=torch.float64),
    )


class _Model:
    """Minimal stand-in exposing the typed ``factors`` sequence."""

    def __init__(self, *factors: object) -> None:
        self.factors = list(factors)


class TestScalarDiagnosticsContract:
    def test_trainable_ee_cusp_reports_constrained_and_raw_ranges(self) -> None:
        cusp = ElectronElectronCusp(trainable_range=True)
        scalars = cusp.scalar_diagnostics()
        assert set(scalars) == {
            "same_range_parameter",
            "opposite_range_parameter",
            "raw_same_range",
            "raw_opposite_range",
        }
        # The softplus makes the two axes different. A trace of raws alone could
        # show motion where the effective parameter has settled, or the reverse.
        assert scalars["same_range_parameter"] != scalars["raw_same_range"]
        assert scalars["same_range_parameter"] > 0.0

    def test_fixed_ee_cusp_reports_no_raw_axis(self) -> None:
        scalars = ElectronElectronCusp(trainable_range=False).scalar_diagnostics()
        assert set(scalars) == {"same_range_parameter", "opposite_range_parameter"}

    def test_trainable_en_law_reports_c_d_and_the_tail_slope_it_implies(self) -> None:
        factor = ElectronNucleusCusp(
            _atoms(),
            law=CurvatureElectronNucleusCuspLaw(
                curvature_coefficient=0.01, curvature_range=1.0, trainable=True
            ),
        )
        scalars = factor.scalar_diagnostics()
        assert scalars["law.curvature_coefficient"] == 0.01
        assert abs(scalars["law.curvature_range"] - 1.0) < 1e-6
        # The tail slope is -Z + c/d, NOT -Z. It comes from the law's own
        # `outer_tail_slope` so a consumer never re-derives it from c and d.
        assert abs(scalars["law.outer_tail_slope.0"] - (-1.99)) < 1e-6

    def test_linear_law_owns_no_scalars(self) -> None:
        factor = ElectronNucleusCusp(_atoms(), law=LinearElectronNucleusCuspLaw())
        assert factor.scalar_diagnostics() == {}


class TestCollectFactorScalars:
    def test_indexes_keep_same_class_factors_distinguishable(self) -> None:
        model = _Model(
            ElectronElectronCusp(trainable_range=True),
            ElectronElectronCusp(trainable_range=True),
        )
        scalars = collect_factor_scalars(model)
        assert "factors.0.same_range_parameter" in scalars
        assert "factors.1.same_range_parameter" in scalars

    def test_production_factor_pipeline_reports_all_four_trainable_scalars(self) -> None:
        model = _Model(
            ElectronElectronCusp(trainable_range=True),
            ElectronNucleusCusp(
                _atoms(),
                law=CurvatureElectronNucleusCuspLaw(
                    curvature_coefficient=0.01, curvature_range=1.0, trainable=True
                ),
            ),
        )
        scalars = collect_factor_scalars(model)
        # These four are exactly the parameters the free 5000-update evidence
        # could NOT speak to, because none of them existed in that run.
        assert "factors.0.raw_same_range" in scalars
        assert "factors.0.raw_opposite_range" in scalars
        assert "factors.1.law.raw_curvature_coefficient" in scalars
        assert "factors.1.law.raw_curvature_range" in scalars

    def test_model_without_factors_reports_nothing_rather_than_raising(self) -> None:
        assert collect_factor_scalars(object()) == {}

    def test_the_trace_moves_when_the_parameter_moves(self) -> None:
        # Without this the suite could not tell a live trace from a constant.
        cusp = ElectronElectronCusp(trainable_range=True)
        before = collect_factor_scalars(_Model(cusp))["factors.0.same_range_parameter"]
        with torch.no_grad():
            cusp.raw_same_range.add_(1.0)
        after = collect_factor_scalars(_Model(cusp))["factors.0.same_range_parameter"]
        assert after > before

    def test_the_zero_coefficient_gradient_trap_is_real_and_the_arm_avoids_it(self) -> None:
        """Why the production config initializes ``c`` nonzero.

        At exactly ``c = 0`` the gradient with respect to the range parameter
        ``d`` is identically zero, so a defaults-instantiated trainable range
        cannot move on step one -- and the law's own defaults are exactly
        ``trainable=True, curvature_coefficient=0.0``.
        """

        distance = torch.tensor([[[1.0]]], dtype=torch.float64)
        charges = torch.tensor([[[2.0]]], dtype=torch.float64)

        trapped = CurvatureElectronNucleusCuspLaw(curvature_coefficient=0.0)
        trapped.value(distance, charges).sum().backward()
        assert float(trapped.raw_curvature_range.grad.abs().item()) == 0.0

        chosen = CurvatureElectronNucleusCuspLaw(curvature_coefficient=0.01)
        chosen.value(distance, charges).sum().backward()
        assert float(chosen.raw_curvature_range.grad.abs().item()) > 0.0
