"""Tests for the tolerance-gated logabs oracle and singlet-purity diagnostic.

`TransformConsistencySummary` is shared by several studies, so the default
(ungated) `failure_count` is a cross-lane contract: it must remain the sign
mismatch count alone. These tests pin that, the opt-in union behavior, and the
triplet-fraction diagnostic including its exclusion semantics.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tpen.data.batch import ElectronBatch
from tpen.evaluation.bundle import EvaluationBundle, GeneratedConfigurations, TransformComparisonValues
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.summaries.trace import DEFAULT_LOGABS_GROSS_ATOL, TransformConsistencySummary

# The spatial-exchange logabs error the post-merge He smoke actually recorded
# alongside `task_success: true`. Any permissive default that fails to count
# this sample would reintroduce exactly the defect this slice repairs.
HE_SMOKE_LOGABS_MAX_ABS_ERROR = 1.1952


def _context(tmp_path: Path) -> EvaluationContext:
    return EvaluationContext(
        namespace="validation/spatial_exchange_symmetry",
        artifact_level="metrics_only",
        task_failure_policy="continue",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=0,
        run_dir=tmp_path,
        task_output_dir=tmp_path,
        metadata={},
    )


def _bundle(
    *,
    original_logabs: list[float],
    transformed_logabs: list[float],
    original_sign: list[float],
    transformed_sign: list[float],
    sign_mismatch: list[bool] | None = None,
) -> EvaluationBundle:
    """Build a bundle carrying only the transform values the summary reads."""

    original = torch.tensor(original_logabs, dtype=torch.float64)
    transformed = torch.tensor(transformed_logabs, dtype=torch.float64)
    signs_original = torch.tensor(original_sign, dtype=torch.float64)
    signs_transformed = torch.tensor(transformed_sign, dtype=torch.float64)
    if sign_mismatch is None:
        # Mirror `SpatialExchangeSymmetryCalculator`: same-sign is expected.
        mismatch = ~torch.isclose(signs_transformed, signs_original, atol=1.0e-6, rtol=1.0e-6)
    else:
        mismatch = torch.tensor(sign_mismatch, dtype=torch.bool)
    batch = ElectronBatch(
        positions=torch.zeros(original.numel(), 2, 3, dtype=torch.float64),
        spins=torch.tensor([[1.0, -1.0]] * original.numel(), dtype=torch.float64),
    )
    return EvaluationBundle(
        generated=GeneratedConfigurations(batch=batch, metadata={}),
        transform=TransformComparisonValues(
            original_logabs=original,
            transformed_logabs=transformed,
            original_sign=signs_original,
            transformed_sign=signs_transformed,
            logabs_abs_error=(transformed - original).abs(),
            sign_mismatch=mismatch,
            metadata={},
        ),
    )


def _metrics(bundle: EvaluationBundle, tmp_path: Path, **kwargs: float | None) -> dict[str, object]:
    return TransformConsistencySummary(**kwargs).summarize(
        bundle=bundle,
        context=_context(tmp_path),
        namespace="validation/spatial_exchange_symmetry",
    ).metrics


def test_sign_only_mismatch_counts_in_both_modes(tmp_path: Path) -> None:
    bundle = _bundle(
        original_logabs=[0.5, 0.5],
        transformed_logabs=[0.5, 0.5],
        original_sign=[1.0, 1.0],
        transformed_sign=[1.0, -1.0],
    )

    ungated = _metrics(bundle, tmp_path)
    gated = _metrics(bundle, tmp_path, logabs_atol=1.0e-6)

    assert ungated["sign_failure_count"] == 1
    assert ungated["logabs_failure_count"] == 0
    assert ungated["failure_count"] == 1
    assert gated["failure_count"] == 1


def test_logabs_only_mismatch_is_reported_but_ungated_by_default(tmp_path: Path) -> None:
    bundle = _bundle(
        original_logabs=[0.0, 0.0],
        transformed_logabs=[0.0, HE_SMOKE_LOGABS_MAX_ABS_ERROR],
        original_sign=[1.0, 1.0],
        transformed_sign=[1.0, 1.0],
    )

    ungated = _metrics(bundle, tmp_path)
    gated = _metrics(bundle, tmp_path, logabs_atol=1.0e-6)

    assert HE_SMOKE_LOGABS_MAX_ABS_ERROR > DEFAULT_LOGABS_GROSS_ATOL
    assert ungated["sign_failure_count"] == 0
    # Always emitted, so the mismatch is never invisible ...
    assert ungated["logabs_failure_count"] == 1
    assert ungated["logabs_tolerance_gated"] is False
    # ... but it does not move `failure_count` unless the lane opted in.
    assert ungated["failure_count"] == 0
    assert gated["logabs_tolerance_gated"] is True
    assert gated["failure_count"] == 1


def test_gated_failure_count_is_the_per_sample_union(tmp_path: Path) -> None:
    # Sample 0 fails on sign only, sample 1 on logabs only, sample 2 on both,
    # sample 3 on neither: the union is 3, not the 2 + 2 of naive addition.
    bundle = _bundle(
        original_logabs=[0.0, 0.0, 0.0, 0.0],
        transformed_logabs=[0.0, 4.0, 4.0, 0.0],
        original_sign=[1.0, 1.0, 1.0, 1.0],
        transformed_sign=[-1.0, 1.0, -1.0, 1.0],
    )

    gated = _metrics(bundle, tmp_path, logabs_atol=1.0e-6)

    assert gated["sign_failure_count"] == 2
    assert gated["logabs_failure_count"] == 2
    assert gated["failure_count"] == 3


def test_rtol_alone_arms_the_gate(tmp_path: Path) -> None:
    bundle = _bundle(
        original_logabs=[10.0],
        transformed_logabs=[12.0],
        original_sign=[1.0],
        transformed_sign=[1.0],
    )

    armed_loose = _metrics(bundle, tmp_path, logabs_rtol=0.5)
    armed_tight = _metrics(bundle, tmp_path, logabs_rtol=0.01)

    # 2.0 <= 0.5 * |10.0| passes; 2.0 <= 0.01 * |10.0| does not.
    assert armed_loose["logabs_tolerance_gated"] is True
    assert armed_loose["logabs_failure_count"] == 0
    assert armed_loose["failure_count"] == 0
    assert armed_tight["failure_count"] == 1


@pytest.mark.parametrize(
    "transformed_logabs",
    [
        [0.0, 0.0, 0.0],  # no logabs error at all
        [0.0, 0.5, 0.9],  # below the gross default
        [0.0, 3.0, 40.0],  # far above the gross default
        [0.0, math.nan, math.inf],  # unevaluable
    ],
)
def test_default_failure_count_is_sign_mismatch_only(tmp_path: Path, transformed_logabs: list[float]) -> None:
    """Pin the cross-lane contract for the three non-He configs' code path.

    `TransformConsistencySummary` with no tolerances is what he-v1
    `full_model_antisymmetry`, hooke `pair_stability_v3` pair_validation,
    `tpen-pair-scan-v1` eval and `tpen-pair-v1` eval all instantiate. For those,
    `failure_count` must equal the sign-mismatch count for ANY logabs error,
    including nonfinite, or this slice would silently flip another lane's
    pass/fail.
    """

    bundle = _bundle(
        original_logabs=[0.0, 0.0, 0.0],
        transformed_logabs=transformed_logabs,
        original_sign=[1.0, 1.0, 1.0],
        transformed_sign=[1.0, -1.0, 1.0],
    )

    metrics = _metrics(bundle, tmp_path)

    sign_failure_count = int(bundle.transform.sign_mismatch.sum().item())
    assert sign_failure_count == 1
    assert metrics["failure_count"] == sign_failure_count
    assert metrics["sign_failure_count"] == sign_failure_count
    assert "logabs_failure_count" in metrics


def test_nonfinite_logabs_error_counts_as_a_logabs_failure(tmp_path: Path) -> None:
    bundle = _bundle(
        original_logabs=[0.0, 0.0],
        transformed_logabs=[0.0, math.nan],
        original_sign=[1.0, 1.0],
        transformed_sign=[1.0, 1.0],
    )

    metrics = _metrics(bundle, tmp_path, logabs_atol=1.0e-6)

    # An unevaluable sample must never be reported as agreement.
    assert metrics["logabs_failure_count"] == 1
    assert metrics["failure_count"] == 1


def test_triplet_fraction_is_zero_for_an_exactly_symmetric_pair(tmp_path: Path) -> None:
    bundle = _bundle(
        original_logabs=[-1.25, 0.75],
        transformed_logabs=[-1.25, 0.75],
        original_sign=[1.0, -1.0],
        transformed_sign=[1.0, -1.0],
    )

    metrics = _metrics(bundle, tmp_path)

    assert metrics["triplet_fraction_mean_under_psi_orig_sq"] == pytest.approx(0.0)
    assert metrics["triplet_fraction_max_under_psi_orig_sq"] == pytest.approx(0.0)
    assert metrics["triplet_fraction_finite_sample_count"] == 2
    assert metrics["triplet_fraction_excluded_sample_count"] == 0


def test_triplet_fraction_is_one_for_an_exactly_antisymmetric_pair(tmp_path: Path) -> None:
    bundle = _bundle(
        original_logabs=[-1.25, 0.75],
        transformed_logabs=[-1.25, 0.75],
        original_sign=[1.0, -1.0],
        transformed_sign=[-1.0, 1.0],
    )

    metrics = _metrics(bundle, tmp_path)

    assert metrics["triplet_fraction_mean_under_psi_orig_sq"] == pytest.approx(1.0)
    assert metrics["triplet_fraction_max_under_psi_orig_sq"] == pytest.approx(1.0)
    assert metrics["triplet_fraction_finite_sample_count"] == 2


def test_triplet_fraction_matches_the_closed_form_for_a_partial_mixture(tmp_path: Path) -> None:
    log_difference = 0.4
    bundle = _bundle(
        original_logabs=[0.0],
        transformed_logabs=[log_difference],
        original_sign=[1.0],
        transformed_sign=[1.0],
    )

    metrics = _metrics(bundle, tmp_path)

    ratio = math.exp(log_difference)
    expected = (1.0 - ratio) ** 2 / ((1.0 - ratio) ** 2 + (1.0 + ratio) ** 2)
    assert 0.0 < expected < 1.0
    assert metrics["triplet_fraction_mean_under_psi_orig_sq"] == pytest.approx(expected)


def test_triplet_fraction_saturates_at_one_half_without_overflow(tmp_path: Path) -> None:
    """A huge log difference must not overflow `exp` into NaN.

    `r -> +-inf` and `r -> 0` both mean the two amplitudes share no support, for
    which `f = 1/2`. Forming `r` directly would produce `inf/inf` here.
    """

    bundle = _bundle(
        original_logabs=[0.0, 0.0],
        transformed_logabs=[900.0, -900.0],
        original_sign=[1.0, 1.0],
        transformed_sign=[1.0, -1.0],
    )

    metrics = _metrics(bundle, tmp_path)

    assert metrics["triplet_fraction_mean_under_psi_orig_sq"] == pytest.approx(0.5)
    assert metrics["triplet_fraction_max_under_psi_orig_sq"] == pytest.approx(0.5)
    assert metrics["triplet_fraction_finite_sample_count"] == 2


def test_nonfinite_and_nodal_samples_are_excluded_not_imputed(tmp_path: Path) -> None:
    """Excluded samples are counted, never folded in as 0.0.

    Imputing 0.0 would report a perfectly pure singlet for a sample whose ratio
    could not be formed at all -- the failure direction that reads as success.
    """

    bundle = _bundle(
        original_logabs=[0.0, math.nan, 0.0, math.inf],
        transformed_logabs=[0.0, 0.0, 0.0, 0.0],
        original_sign=[1.0, 1.0, 0.0, 1.0],
        transformed_sign=[-1.0, 1.0, 1.0, 1.0],
        sign_mismatch=[True, False, False, False],
    )

    metrics = _metrics(bundle, tmp_path)

    # Only sample 0 is usable, and it is a pure triplet.
    assert metrics["triplet_fraction_finite_sample_count"] == 1
    assert metrics["triplet_fraction_excluded_sample_count"] == 3
    assert metrics["triplet_fraction_mean_under_psi_orig_sq"] == pytest.approx(1.0)


def test_triplet_fraction_metrics_are_nan_when_every_sample_is_excluded(tmp_path: Path) -> None:
    bundle = _bundle(
        original_logabs=[math.nan],
        transformed_logabs=[math.nan],
        original_sign=[1.0],
        transformed_sign=[1.0],
    )

    metrics = _metrics(bundle, tmp_path)

    assert math.isnan(float(metrics["triplet_fraction_mean_under_psi_orig_sq"]))
    assert math.isnan(float(metrics["triplet_fraction_max_under_psi_orig_sq"]))
    assert metrics["triplet_fraction_finite_sample_count"] == 0
    assert metrics["triplet_fraction_excluded_sample_count"] == 1
