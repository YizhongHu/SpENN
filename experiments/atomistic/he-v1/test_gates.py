"""Gate behavior on the He-v1 cusp and tail metrics.

Each test here guards a way a receipt could come back green while saying nothing.
The failures are not hypothetical: the post-merge He smoke packet extracted
``cusp_available`` and the ``*_count`` keys and never read the slopes that
``tpen/evaluation/summaries/atom.py`` had already emitted, so "the gates passed"
and "the physics was checked" came apart once already.

unavailable is not zero
    An unfitted region must report ``absent`` with a value of ``None``. A zero
    slope parses as data and drags any median over rows toward zero.

counts survive the values
    ``*_available`` and ``*_count`` are gated in their own right, so a row still
    carries how much data produced its means.

the charge owns -Z
    The expected-slope target comes from ``spec['nuclear_charge']``. If a fitted
    value could supply the reference, the cusp gate would compare a number
    against itself and always pass.

an unset threshold is not a pass
    Before the tolerances are predeclared, every value gate must read ``absent``,
    never ``pass``. A mistyped spec key raises instead of quietly disabling the
    one gate it was meant to set.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

STUDY_DIR = Path(__file__).resolve().parent


def _load_study_module(name: str) -> ModuleType:
    """Load one study module by path, under its own module name.

    The study directory is not an importable package (its name contains a
    hyphen), so the checked-in experiment tests load their subjects by file
    location rather than by import.
    """

    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"he_v1_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gates = _load_study_module("gates")
evaluate_atom_gates = gates.evaluate_atom_gates
GateOutcome = gates.GateOutcome

CHARGE = 2.0
EXPECTED_SLOPE = -CHARGE

# A fully available, physically sane He row: cusp slopes sit on -Z and every
# outer ray decays. Individual tests perturb exactly one field.
HEALTHY_METRICS: dict[str, Any] = {
    "cusp_available": True,
    "cusp_finite_fit_count": 6,
    "cusp_finite_measurement_count": 84,
    "cusp_total_measurement_count": 84,
    "cusp_expected_slope": EXPECTED_SLOPE,
    "cusp_one_sided_slope_mean": -1.98,
    "cusp_one_sided_slope_abs_error_mean": 0.02,
    "cusp_one_sided_slope_abs_error_max": 0.05,
    "tail_available": True,
    "tail_outer_measurement_count": 6,
    "tail_finite_measurement_count": 84,
    "tail_total_measurement_count": 84,
    "tail_outer_slope_mean": -2.6,
    "tail_outer_slope_min": -3.1,
    "tail_outer_slope_max": -2.2,
    "tail_outer_radius_min": 4.0,
    "tail_outer_radius_max": 6.0,
    "tail_negative_slope_fraction": 1.0,
}

# Tolerances loose enough that HEALTHY_METRICS passes every gate. These are test
# fixtures, not the production numbers: the He-v1 tolerances are predeclared
# separately, before the run.
PERMISSIVE_SPEC: dict[str, Any] = {
    "nuclear_charge": CHARGE,
    "require_cusp_available": True,
    "cusp_finite_fit_count_min": 4,
    "cusp_finite_measurement_count_min": 40,
    "cusp_expected_slope_tolerance": 1e-9,
    "cusp_one_sided_slope_mean_abs_error_max": 0.1,
    "cusp_one_sided_slope_abs_error_mean_max": 0.1,
    "cusp_one_sided_slope_abs_error_max_max": 0.2,
    "require_tail_available": True,
    "tail_outer_measurement_count_min": 4,
    "tail_finite_measurement_count_min": 40,
    "tail_negative_slope_fraction_min": 1.0,
    "tail_outer_slope_max_max": -0.5,
    "tail_outer_slope_mean_max": -1.0,
    "tail_outer_slope_mean_min": -8.0,
    "tail_outer_slope_min_min": -8.0,
    "tail_outer_radius_min_min": 3.0,
    "tail_outer_radius_max_max": 12.0,
}


def _by_name(outcomes: tuple[GateOutcome, ...]) -> dict[str, GateOutcome]:
    """Index outcomes by gate name, asserting the names are unique."""

    indexed = {outcome.name: outcome for outcome in outcomes}
    assert len(indexed) == len(outcomes), "gate names must be unique within one evaluation"
    return indexed


def _evaluate(
    *,
    metrics: dict[str, Any] | None = None,
    spec: dict[str, Any] | None = None,
) -> dict[str, GateOutcome]:
    """Evaluate the healthy row with optional overrides, indexed by name."""

    return _by_name(
        evaluate_atom_gates(
            dict(HEALTHY_METRICS) | (metrics or {}),
            spec=dict(PERMISSIVE_SPEC) | (spec or {}),
        )
    )


class TestHealthyRow:
    """A sane, fully available row gates green on everything."""

    def test_every_gate_passes(self) -> None:
        outcomes = evaluate_atom_gates(HEALTHY_METRICS, spec=PERMISSIVE_SPEC)
        assert outcomes, "the gate set must not be empty"
        offenders = {
            outcome.name: outcome.reason for outcome in outcomes if outcome.status != "pass"
        }
        assert offenders == {}

    def test_every_outcome_explains_itself(self) -> None:
        # A receipt row with an empty reason is unreadable months later, and a
        # blank reason on `absent` is exactly the case that needs the words.
        for outcome in evaluate_atom_gates(HEALTHY_METRICS, spec=PERMISSIVE_SPEC):
            assert outcome.reason.strip(), f"{outcome.name} has no reason"

    def test_outcomes_are_a_tuple_of_frozen_records(self) -> None:
        outcomes = evaluate_atom_gates(HEALTHY_METRICS, spec=PERMISSIVE_SPEC)
        assert isinstance(outcomes, tuple)
        with pytest.raises(Exception):
            outcomes[0].status = "pass"  # type: ignore[misc]

    def test_status_vocabulary_is_closed(self) -> None:
        for outcome in evaluate_atom_gates(HEALTHY_METRICS, spec=PERMISSIVE_SPEC):
            assert outcome.status in {"pass", "fail", "absent"}


class TestValuesAreActuallyRead:
    """The emitted numbers reach the gates, rather than the flags standing in."""

    @pytest.mark.parametrize("metric", sorted(gates.ATOM_GATE_METRIC_KEYS))
    def test_every_declared_metric_key_is_gated(self, metric: str) -> None:
        # Guards the original defect: a metric that atom.py emits but nothing
        # reads. Every key in the module's declared read-set must own an outcome
        # whose value came from the metrics mapping.
        outcomes = evaluate_atom_gates(HEALTHY_METRICS, spec=PERMISSIVE_SPEC)
        touching = [
            outcome for outcome in outcomes if metric in outcome.reason and outcome.value is not None
        ]
        assert touching, f"no gate reads '{metric}'"

    def test_the_four_cusp_and_six_tail_values_are_all_covered(self) -> None:
        # Named explicitly rather than derived, so dropping one from the
        # definition table is a test failure and not a silently smaller gate set.
        required = {
            "cusp_expected_slope",
            "cusp_one_sided_slope_mean",
            "cusp_one_sided_slope_abs_error_mean",
            "cusp_one_sided_slope_abs_error_max",
            "tail_outer_slope_mean",
            "tail_outer_slope_min",
            "tail_outer_slope_max",
            "tail_outer_radius_min",
            "tail_outer_radius_max",
            "tail_negative_slope_fraction",
        }
        assert required <= set(gates.ATOM_GATE_METRIC_KEYS)

    def test_reported_value_is_the_metric_value(self) -> None:
        outcomes = _evaluate(metrics={"cusp_one_sided_slope_abs_error_mean": 0.031})
        assert outcomes["cusp_one_sided_slope_abs_error_mean_at_most"].value == pytest.approx(0.031)


class TestGateFailures:
    """Each bound actually bites."""

    @pytest.mark.parametrize(
        ("gate", "metric", "value"),
        [
            ("cusp_one_sided_slope_abs_error_mean_at_most", "cusp_one_sided_slope_abs_error_mean", 0.5),
            ("cusp_one_sided_slope_abs_error_max_at_most", "cusp_one_sided_slope_abs_error_max", 0.5),
            ("cusp_one_sided_slope_mean_near_charge", "cusp_one_sided_slope_mean", -1.2),
            ("cusp_expected_slope_near_charge", "cusp_expected_slope", -3.0),
            ("tail_negative_slope_fraction_at_least", "tail_negative_slope_fraction", 0.5),
            ("tail_outer_slope_max_at_most", "tail_outer_slope_max", 0.4),
            ("tail_outer_slope_mean_at_most", "tail_outer_slope_mean", -0.2),
            ("tail_outer_slope_mean_at_least", "tail_outer_slope_mean", -40.0),
            ("tail_outer_slope_min_at_least", "tail_outer_slope_min", -40.0),
            ("tail_outer_radius_min_at_least", "tail_outer_radius_min", 0.5),
            ("tail_outer_radius_max_at_most", "tail_outer_radius_max", 40.0),
            ("cusp_finite_fit_count_at_least", "cusp_finite_fit_count", 1),
            ("tail_outer_measurement_count_at_least", "tail_outer_measurement_count", 1),
        ],
    )
    def test_out_of_bound_value_fails(self, gate: str, metric: str, value: float) -> None:
        outcomes = _evaluate(metrics={metric: value})
        assert outcomes[gate].status == "fail", outcomes[gate].reason
        assert outcomes[gate].value == pytest.approx(value)
        assert outcomes[gate].threshold is not None

    def test_a_growing_tail_is_caught_by_sign_not_only_by_magnitude(self) -> None:
        # A positive outer slope means the amplitude grows outward. This is the
        # single most important tail failure and two independent gates see it.
        outcomes = _evaluate(
            metrics={
                "tail_outer_slope_max": 0.7,
                "tail_negative_slope_fraction": 0.5,
            }
        )
        assert outcomes["tail_outer_slope_max_at_most"].status == "fail"
        assert outcomes["tail_negative_slope_fraction_at_least"].status == "fail"

    def test_boundary_value_passes(self) -> None:
        # The comparisons are inclusive; a value exactly on its declared bound is
        # within tolerance, not outside it.
        outcomes = _evaluate(metrics={"cusp_one_sided_slope_abs_error_mean": 0.1})
        assert outcomes["cusp_one_sided_slope_abs_error_mean_at_most"].status == "pass"

    def test_non_finite_value_fails_rather_than_passing_a_nan_comparison(self) -> None:
        # `nan <= t` is False, so this would land as a failure by accident. It is
        # asserted deliberately: a NaN slope is corrupt data, not missing data.
        outcomes = _evaluate(metrics={"tail_outer_slope_mean": float("nan")})
        assert outcomes["tail_outer_slope_mean_at_most"].status == "fail"
        assert "finite" in outcomes["tail_outer_slope_mean_at_most"].reason

    def test_non_numeric_value_fails(self) -> None:
        outcomes = _evaluate(metrics={"tail_outer_slope_mean": "-2.6"})
        assert outcomes["tail_outer_slope_mean_at_most"].status == "fail"

    def test_a_boolean_in_a_numeric_slot_does_not_compare_as_one(self) -> None:
        # bool subclasses int, so True would otherwise gate as the number 1.
        outcomes = _evaluate(metrics={"cusp_one_sided_slope_abs_error_mean": True})
        assert outcomes["cusp_one_sided_slope_abs_error_mean_at_most"].status == "fail"


class TestAbsentIsNeverZero:
    """An unavailable fit reports absence, and reports why."""

    def test_unavailable_cusp_yields_absent_value_gates_with_reasons(self) -> None:
        outcomes = _evaluate(
            metrics={
                "cusp_available": False,
                "cusp_finite_fit_count": 0,
                "cusp_finite_measurement_count": 0,
                # The stale value keys are deliberately LEFT in place. atom.py
                # omits them when nothing is fitted, but a collector that merged
                # a previous row would carry them forward, and the flag must win
                # over them.
            }
        )
        for gate in (
            "cusp_expected_slope_near_charge",
            "cusp_one_sided_slope_mean_near_charge",
            "cusp_one_sided_slope_abs_error_mean_at_most",
            "cusp_one_sided_slope_abs_error_max_at_most",
        ):
            assert outcomes[gate].status == "absent"
            assert outcomes[gate].value is None, "an unavailable fit must not carry a value"
            assert "cusp_available" in outcomes[gate].reason

    def test_absent_value_is_none_not_zero(self) -> None:
        metrics = {
            key: value
            for key, value in HEALTHY_METRICS.items()
            if not key.startswith("tail_outer_slope") and key != "tail_negative_slope_fraction"
        }
        metrics["tail_available"] = False
        metrics["tail_outer_measurement_count"] = 0
        outcomes = _by_name(evaluate_atom_gates(metrics, spec=PERMISSIVE_SPEC))
        slope_gate = outcomes["tail_outer_slope_mean_at_most"]
        assert slope_gate.status == "absent"
        assert slope_gate.value is None
        assert slope_gate.value != 0.0

    def test_missing_metric_is_absent_not_defaulted(self) -> None:
        metrics = dict(HEALTHY_METRICS)
        del metrics["tail_outer_radius_min"]
        outcomes = _by_name(evaluate_atom_gates(metrics, spec=PERMISSIVE_SPEC))
        gate = outcomes["tail_outer_radius_min_at_least"]
        assert gate.status == "absent"
        assert gate.value is None
        assert "tail_outer_radius_min" in gate.reason

    def test_missing_availability_flag_is_absent_not_assumed_available(self) -> None:
        metrics = dict(HEALTHY_METRICS)
        del metrics["cusp_available"]
        outcomes = _by_name(evaluate_atom_gates(metrics, spec=PERMISSIVE_SPEC))
        assert outcomes["cusp_expected_slope_near_charge"].status == "absent"
        assert outcomes["cusp_available"].status == "absent"

    def test_required_availability_flag_false_is_a_failure_not_a_pass(self) -> None:
        # The value gates go absent, but the row must still be visibly bad:
        # absence everywhere with no failure anywhere reads as success.
        outcomes = _evaluate(metrics={"cusp_available": False})
        assert outcomes["cusp_available"].status == "fail"
        assert outcomes["cusp_available"].value is False

    def test_unavailable_and_not_required_is_absent(self) -> None:
        outcomes = _evaluate(
            metrics={"tail_available": False},
            spec={"require_tail_available": False},
        )
        assert outcomes["tail_available"].status == "absent"


class TestCountsAreRetained:
    """Availability and counts survive alongside the values, not instead of them."""

    def test_counts_are_reported_even_on_a_fully_healthy_row(self) -> None:
        outcomes = _evaluate()
        assert outcomes["cusp_finite_fit_count_at_least"].value == pytest.approx(6)
        assert outcomes["tail_outer_measurement_count_at_least"].value == pytest.approx(6)
        assert outcomes["cusp_finite_measurement_count_at_least"].value == pytest.approx(84)
        assert outcomes["tail_finite_measurement_count_at_least"].value == pytest.approx(84)

    def test_counts_remain_readable_when_the_fit_is_unavailable(self) -> None:
        # The whole point of retention: an unavailable row must still say how
        # many measurements were attempted and how many were finite.
        outcomes = _evaluate(
            metrics={
                "cusp_available": False,
                "cusp_finite_fit_count": 0,
                "cusp_finite_measurement_count": 0,
            }
        )
        assert outcomes["cusp_finite_fit_count_at_least"].value == pytest.approx(0)
        assert outcomes["cusp_finite_fit_count_at_least"].status == "fail"
        assert outcomes["cusp_finite_measurement_count_at_least"].status == "fail"

    def test_availability_flags_have_their_own_outcomes(self) -> None:
        outcomes = _evaluate()
        assert outcomes["cusp_available"].value is True
        assert outcomes["tail_available"].value is True

    def test_non_boolean_availability_flag_fails(self) -> None:
        outcomes = _evaluate(metrics={"cusp_available": 1})
        assert outcomes["cusp_available"].status == "fail"


class TestChargeOwnsTheExpectedSlope:
    """``-Z`` comes from the spec, never from a fit."""

    def test_target_follows_the_declared_charge(self) -> None:
        # Same fitted slope, different declared charge: the verdict must flip.
        # If the fit supplied its own reference this test could not fail.
        metrics = {"cusp_one_sided_slope_mean": -1.98, "cusp_expected_slope": -2.0}
        at_z2 = _evaluate(metrics=metrics, spec={"nuclear_charge": 2.0})
        at_z3 = _evaluate(metrics=metrics, spec={"nuclear_charge": 3.0})
        assert at_z2["cusp_one_sided_slope_mean_near_charge"].status == "pass"
        assert at_z3["cusp_one_sided_slope_mean_near_charge"].status == "fail"

    def test_a_drifted_expected_slope_is_itself_caught(self) -> None:
        # atom.py's `cusp_expected_slope` is a reported quantity, not an axiom.
        # If it stops equaling -Z the summary is describing a different atom.
        outcomes = _evaluate(metrics={"cusp_expected_slope": -1.0})
        gate = outcomes["cusp_expected_slope_near_charge"]
        assert gate.status == "fail"
        assert "-2.0" in gate.reason

    def test_reason_names_the_charge_owned_target(self) -> None:
        gate = _evaluate()["cusp_one_sided_slope_mean_near_charge"]
        assert "-Z" in gate.reason

    def test_missing_charge_leaves_the_cusp_gates_undecided(self) -> None:
        spec = {key: value for key, value in PERMISSIVE_SPEC.items() if key != "nuclear_charge"}
        outcomes = _by_name(evaluate_atom_gates(HEALTHY_METRICS, spec=spec))
        assert outcomes["cusp_expected_slope_near_charge"].status == "absent"
        assert outcomes["cusp_one_sided_slope_mean_near_charge"].status == "absent"

    @pytest.mark.parametrize("charge", [0.0, -2.0, float("nan"), "2", True])
    def test_an_unusable_charge_raises_rather_than_gating(self, charge: Any) -> None:
        with pytest.raises(ValueError):
            evaluate_atom_gates(HEALTHY_METRICS, spec=dict(PERMISSIVE_SPEC) | {"nuclear_charge": charge})


class TestThresholdsAreParameters:
    """No tolerance is a literal, and an unset one never reads as a pass."""

    def test_an_empty_spec_decides_nothing(self) -> None:
        outcomes = evaluate_atom_gates(HEALTHY_METRICS, spec={})
        assert {outcome.status for outcome in outcomes} == {"absent"}

    def test_an_undeclared_threshold_retains_the_value_ungated(self) -> None:
        spec = {
            key: value
            for key, value in PERMISSIVE_SPEC.items()
            if key != "tail_outer_slope_mean_max"
        }
        outcomes = _by_name(evaluate_atom_gates(HEALTHY_METRICS, spec=spec))
        gate = outcomes["tail_outer_slope_mean_at_most"]
        assert gate.status == "absent"
        assert gate.value == pytest.approx(-2.6), "the observed value must survive"
        assert gate.threshold is None
        # The sibling gate over the same metric is still decided.
        assert outcomes["tail_outer_slope_mean_at_least"].status == "pass"

    def test_tightening_a_threshold_flips_a_passing_gate(self) -> None:
        # Proves the threshold reaches the comparison instead of a constant.
        outcomes = _evaluate(spec={"cusp_one_sided_slope_abs_error_mean_max": 0.001})
        assert outcomes["cusp_one_sided_slope_abs_error_mean_at_most"].status == "fail"
        assert outcomes["cusp_one_sided_slope_abs_error_mean_at_most"].threshold == pytest.approx(0.001)

    def test_an_unknown_spec_key_raises(self) -> None:
        # A mistyped tolerance would otherwise disable exactly the gate it was
        # written to set, and disable it silently.
        with pytest.raises(ValueError, match="unknown atom gate spec keys"):
            evaluate_atom_gates(
                HEALTHY_METRICS,
                spec=dict(PERMISSIVE_SPEC) | {"tail_outer_slope_mean_maximum": -1.0},
            )

    def test_every_declared_spec_key_is_consumed_by_a_gate(self) -> None:
        assert set(PERMISSIVE_SPEC) == set(gates.ATOM_GATE_SPEC_KEYS)

    @pytest.mark.parametrize("threshold", ["0.1", None])
    def test_a_non_numeric_threshold_is_not_silently_accepted(self, threshold: Any) -> None:
        spec = dict(PERMISSIVE_SPEC) | {"cusp_one_sided_slope_abs_error_mean_max": threshold}
        if threshold is None:
            outcomes = _by_name(evaluate_atom_gates(HEALTHY_METRICS, spec=spec))
            assert outcomes["cusp_one_sided_slope_abs_error_mean_at_most"].status == "absent"
        else:
            with pytest.raises(ValueError):
                evaluate_atom_gates(HEALTHY_METRICS, spec=spec)


class TestStdlibOnly:
    """The module must stay importable without torch or tpen."""

    def test_module_imports_no_tpen(self) -> None:
        source = (STUDY_DIR / "gates.py").read_text(encoding="utf-8")
        offenders = [
            line
            for line in source.splitlines()
            if line.startswith(("import tpen", "from tpen"))
            or line.strip().startswith(("import tpen", "from tpen"))
        ]
        assert offenders == [], "experiments/README.md forbids tpen imports outside launchers"

    def test_module_imports_no_third_party_packages(self) -> None:
        source = (STUDY_DIR / "gates.py").read_text(encoding="utf-8")
        third_party = ("torch", "numpy", "omegaconf", "pandas", "yaml")
        for name in third_party:
            assert f"import {name}" not in source
