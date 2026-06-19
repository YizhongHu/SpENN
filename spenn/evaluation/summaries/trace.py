"""Summaries for transform and trace evaluation records."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from spenn.evaluation.bundle import EvaluationBundle
from spenn.evaluation.protocols import EvaluationContext
from spenn.evaluation.results import MetricScalar, SummaryResult


class TransformConsistencySummary:
    """Summarize model-output consistency under a generated transform."""

    name = "transform_consistency"
    required_fields = frozenset({"transform"})

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return logabs, sign, and optional local-energy error metrics."""

        del context, namespace
        transform = bundle.transform
        if transform is None:
            raise ValueError("TransformConsistencySummary requires bundle.transform")
        logabs_error = _finite_or_empty(transform.logabs_abs_error)
        metrics: dict[str, MetricScalar] = {
            "logabs_max_abs_error": _max(logabs_error),
            "logabs_mean_abs_error": _mean(logabs_error),
            "sign_failure_count": int(transform.sign_mismatch.sum().item()),
            "failure_count": int(transform.sign_mismatch.sum().item()),
        }
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
        return SummaryResult(
            metrics={
                "max_abs_error": _max(finite),
                "mean_abs_error": _mean(finite),
                "failure_count": int(values.failure_count),
                "compared_entry_count": int(values.compared_entry_count),
                "comparison_error_count": int(values.comparison_error_count),
                "missing_key_count": int(values.missing_key_count),
                "extra_key_count": int(values.extra_key_count),
            }
        )


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
    "FeatureTraceSummary",
    "ReadoutTraceSummary",
    "TraceEquivarianceSummary",
    "TransformConsistencySummary",
]
