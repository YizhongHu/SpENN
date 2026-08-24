"""Summaries for transform and trace evaluation records."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from tpen.evaluation.bundle import (
    EvaluationBundle,
    TransformComparisonValues,
    TransformKind,
    TransformName,
)
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.results import MetricScalar, SummaryResult


DEFAULT_LOGABS_GROSS_ATOL = 1.0
"""Permissive absolute logabs tolerance used when no tolerance is configured.

One nat: the transformed amplitude differs from the original by a factor of
``e`` or more. This is deliberately far too loose to be a physics gate. It
exists so ``logabs_failure_count`` is always populated with something that
cannot silently pass a gross mismatch (the He smoke's 1.1952 exceeds it), while
``failure_count`` stays exactly what it was for every config that has not
opted in to logabs gating.
"""


class TransformConsistencySummary:
    """Summarize model-output consistency under a generated transform.

    Parameters
    ----------
    logabs_atol, logabs_rtol : float, optional
        Tolerances for the logabs half of the transform oracle. Both default to
        ``None``, meaning *not gated*: ``logabs_failure_count`` is still emitted
        (counted against `DEFAULT_LOGABS_GROSS_ATOL`) but ``failure_count``
        remains the sign-mismatch count alone, byte-identical to the behavior
        every existing config sees. When either tolerance is set, the logabs
        gate is armed and ``failure_count`` becomes the per-sample UNION of sign
        and logabs mismatch. Opt-in rather than fail-closed because this summary
        is shared across several studies; flipping the default would change
        another lane's pass/fail without its consent.

    Notes
    -----
    The summary also reports the singlet-purity diagnostic. Writing
    ``r = Psi_transformed / Psi_original = (s_t / s_o) exp(L_t - L_o)``, the
    per-sample triplet fraction is
    ``f = |1 - r|^2 / (|1 - r|^2 + |1 + r|^2)``, which is 0 exactly at ``r = 1``
    (pure singlet on the sampled support) and 1 at ``r = -1``. For a spatial
    exchange of an opposite-spin pair at fixed spin labels this measures
    singlet/triplet(M_S=0) mixing of the sampled state -- a convergence and
    state-purity diagnostic, not an architectural invariant. It costs no extra
    model evaluation because the calculator already retains both logabs and
    both signs.
    """

    name = "transform_consistency"
    required_fields = frozenset({"transform"})

    def __init__(
        self,
        *,
        logabs_atol: float | None = None,
        logabs_rtol: float | None = None,
    ) -> None:
        self.logabs_atol = None if logabs_atol is None else float(logabs_atol)
        self.logabs_rtol = None if logabs_rtol is None else float(logabs_rtol)

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return logabs, sign, purity, and optional local-energy error metrics."""

        del context
        transform = bundle.transform
        if transform is None:
            raise ValueError("TransformConsistencySummary requires bundle.transform")
        logabs_error = _finite_or_empty(transform.logabs_abs_error)
        gated = self.logabs_atol is not None or self.logabs_rtol is not None
        if gated:
            atol = 0.0 if self.logabs_atol is None else self.logabs_atol
            rtol = 0.0 if self.logabs_rtol is None else self.logabs_rtol
        else:
            # Ungated: report the count against the documented gross default,
            # but leave `failure_count` alone.
            atol = DEFAULT_LOGABS_GROSS_ATOL
            rtol = 0.0
        logabs_mismatch = _logabs_mismatch(transform, atol=atol, rtol=rtol)
        sign_mismatch = transform.sign_mismatch.detach().reshape(-1)
        failure_mismatch = sign_mismatch | logabs_mismatch if gated else sign_mismatch
        metrics: dict[str, MetricScalar] = {
            "logabs_max_abs_error": _max(logabs_error),
            "logabs_mean_abs_error": _mean(logabs_error),
            "sign_failure_count": int(transform.sign_mismatch.sum().item()),
            "logabs_failure_count": int(logabs_mismatch.sum().item()),
            "logabs_tolerance_gated": gated,
            "logabs_gate_atol": float(atol),
            "logabs_gate_rtol": float(rtol),
            "failure_count": int(failure_mismatch.sum().item()),
        }
        if _spatial_exchange_namespace(transform, namespace):
            metrics.update(_singlet_purity_metrics(transform))
        if transform.finite is not None:
            finite = transform.finite.detach().reshape(-1)
            metrics["finite_count"] = int(finite.sum().item())
            metrics["nonfinite_count"] = int((~finite).sum().item())
        if transform.local_energy_abs_error is not None:
            local_energy_error = _finite_or_empty(transform.local_energy_abs_error)
            metrics["local_energy_max_abs_error"] = _max(local_energy_error)
            metrics["local_energy_mean_abs_error"] = _mean(local_energy_error)
        return SummaryResult(metrics=metrics)


class TraceEquivarianceSummary:
    """Summarize typed trace equivariance comparison records."""

    name = "trace_equivariance"
    required_fields = frozenset({"trace_comparison"})

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return aggregate trace equivariance errors and counts."""

        del context, namespace
        values = bundle.trace_comparison
        if values is None:
            raise ValueError("TraceEquivarianceSummary requires bundle.trace_comparison")
        finite = _finite_or_empty(values.max_abs_error)
        metrics: dict[str, MetricScalar] = {
                "max_abs_error": _max(finite),
                "mean_abs_error": _mean(finite),
                "failure_count": int(values.failure_count),
                "compared_entry_count": int(values.compared_entry_count),
                "compared_sample_count": int(values.compared_sample_count),
                "comparison_error_count": int(values.comparison_error_count),
                "missing_key_count": int(values.missing_key_count),
                "extra_key_count": int(values.extra_key_count),
            }
        for key_summary in values.key_summaries:
            prefix = f"key/{key_summary.key}"
            metrics[f"{prefix}/count"] = int(key_summary.count)
            metrics[f"{prefix}/mean_abs_error"] = key_summary.mean_abs_error
            metrics[f"{prefix}/max_abs_error"] = key_summary.max_abs_error
            metrics[f"{prefix}/failure_count"] = int(key_summary.failure_count)
            metrics[f"{prefix}/missing_key_count"] = int(key_summary.missing_count)
            metrics[f"{prefix}/extra_key_count"] = int(key_summary.extra_count)
            metrics[f"{prefix}/sample_count"] = int(key_summary.sample_count)
        return SummaryResult(metrics=metrics)


class FeatureTraceSummary:
    """Summarize feature-trace magnitude records."""

    name = "feature_trace"
    required_fields = frozenset({"feature_trace"})

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return max/q95 feature magnitudes and nonfinite counts."""

        del context, namespace
        values = bundle.feature_trace
        if values is None:
            raise ValueError("FeatureTraceSummary requires bundle.feature_trace")
        records = values.records
        return SummaryResult(
            metrics={
                "feature_rms_max": _record_max(records, "rms"),
                "feature_rms_q95": _record_quantile(records, "rms", 0.95),
                "feature_max_abs_max": _record_max(records, "max_abs"),
                "feature_nonfinite_count": int(sum(int(record.get("nonfinite_count", 0)) for record in records)),
            }
        )


class ReadoutTraceSummary:
    """Summarize readout/Pfaffian conditioning records."""

    name = "readout_trace"
    required_fields = frozenset({"readout_trace"})

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return readout conditioning and near-zero metrics."""

        del context, namespace
        values = bundle.readout_trace
        if values is None:
            raise ValueError("ReadoutTraceSummary requires bundle.readout_trace")
        records = values.records
        return SummaryResult(
            metrics={
                "pfaffian_near_zero_count": int(sum(int(record.get("near_zero_count", 0)) for record in records)),
                "condition_number_q95": _record_quantile(records, "condition_number", 0.95),
                "condition_number_max": _record_max(records, "condition_number"),
                "readout_nonfinite_count": int(
                    sum(1 for record in records if float(record.get("finite_fraction", 1.0)) < 1.0)
                ),
            }
        )


def _logabs_mismatch(values: TransformComparisonValues, *, atol: float, rtol: float) -> torch.Tensor:
    """Return the per-sample logabs mismatch mask for one tolerance pair.

    The comparison is ``|L_t - L_o| <= atol + rtol * |L_o|``, negated. It is
    written as a negation so that a nonfinite error or a nonfinite reference
    compares false and is counted as a mismatch: a sample whose amplitude ratio
    could not be evaluated must never be reported as agreement.
    """

    error = values.logabs_abs_error.detach().reshape(-1)
    reference = values.original_logabs.detach().reshape(-1).abs()
    return ~(error <= atol + rtol * reference)


def _spatial_exchange_namespace(values: TransformComparisonValues, namespace: str) -> bool:
    """Require typed spatial identity and its explicit namespace for purity."""

    namespace_name = namespace.rstrip("/").split("/")[-1]
    return (
        values.transform_kind == TransformKind.SPATIAL_EXCHANGE
        and namespace_name == TransformName.SPATIAL_EXCHANGE_SYMMETRY.value
    )


def _singlet_purity_metrics(values: TransformComparisonValues) -> dict[str, MetricScalar]:
    """Return triplet-fraction metrics from retained logabs and sign values.

    With ``u = L_t - L_o`` and ``s = sign(s_o s_t) = +-1`` the ratio is
    ``r = s exp(u)``, and the triplet fraction collapses to the closed form

    ``f = |1 - r|^2 / (|1 - r|^2 + |1 + r|^2) = (1 - s sech(u)) / 2``

    which is evaluated directly. That form is what makes the diagnostic
    numerically safe: ``sech`` is computed from ``exp(-|u|)`` so it underflows
    smoothly to 0 for large ``|u|`` (giving ``f = 1/2``, the no-overlap value)
    instead of overflowing ``exp(u)`` to infinity. ``f = 0`` iff ``r = 1`` and
    ``f = 1`` iff ``r = -1``.

    Samples with a nonfinite log difference, a nonfinite sign, or a vanishing
    sign product (where the ratio is undefined) are EXCLUDED and counted, never
    imputed as 0.0. Samples are drawn under ``|Psi_orig|^2``, which the metric
    names state explicitly.
    """

    original_logabs = values.original_logabs.detach().reshape(-1).to(torch.float64)
    transformed_logabs = values.transformed_logabs.detach().reshape(-1).to(torch.float64)
    original_sign = values.original_sign.detach().reshape(-1).to(torch.float64)
    transformed_sign = values.transformed_sign.detach().reshape(-1).to(torch.float64)
    log_difference = transformed_logabs - original_logabs
    sign_ratio = torch.sign(original_sign * transformed_sign)
    usable = torch.isfinite(log_difference) & torch.isfinite(sign_ratio) & (sign_ratio != 0.0)
    # Neutralize excluded entries before the exponential so their NaNs cannot
    # contaminate the finite entries; they are dropped by the mask afterwards.
    magnitude = torch.where(usable, log_difference.abs(), torch.zeros_like(log_difference))
    decay = torch.exp(-magnitude)
    sech = 2.0 * decay / (1.0 + decay * decay)
    triplet_fraction = (0.5 * (1.0 - sign_ratio * sech))[usable]
    # f is analytically in [0, 1]; clamp only removes rounding-scale excursions.
    triplet_fraction = triplet_fraction.clamp(min=0.0, max=1.0)
    return {
        "triplet_fraction_mean_under_psi_orig_sq": _mean(triplet_fraction),
        "triplet_fraction_max_under_psi_orig_sq": _max(triplet_fraction),
        "triplet_fraction_finite_sample_count": int(triplet_fraction.numel()),
        "triplet_fraction_excluded_sample_count": int((~usable).sum().item()),
    }


def _finite_or_empty(values: torch.Tensor) -> torch.Tensor:
    flat = values.detach().reshape(-1)
    return flat[torch.isfinite(flat)]


def _max(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return math.nan
    return float(values.max().item())


def _mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return math.nan
    return float(values.mean().item())


def _record_values(records: Sequence[dict[str, Any]] | Sequence[Any], key: str) -> torch.Tensor:
    values = []
    for record in records:
        if isinstance(record, dict) and key in record:
            value = record[key]
            if isinstance(value, (int, float)):
                values.append(float(value))
    if not values:
        return torch.empty(0, dtype=torch.float64)
    tensor = torch.tensor(values, dtype=torch.float64)
    return tensor[torch.isfinite(tensor)]


def _record_max(records: Sequence[dict[str, Any]] | Sequence[Any], key: str) -> float:
    return _max(_record_values(records, key))


def _record_quantile(records: Sequence[dict[str, Any]] | Sequence[Any], key: str, q: float) -> float:
    values = _record_values(records, key)
    if values.numel() == 0:
        return math.nan
    return float(torch.quantile(values, torch.tensor(float(q), dtype=values.dtype)).item())


__all__ = [
    "DEFAULT_LOGABS_GROSS_ATOL",
    "FeatureTraceSummary",
    "ReadoutTraceSummary",
    "TraceEquivarianceSummary",
    "TransformConsistencySummary",
]
