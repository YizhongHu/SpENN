"""Metadata summaries for evaluation generators."""

from __future__ import annotations

from collections.abc import Mapping

from spenn.evaluation.bundle import EvaluationBundle
from spenn.evaluation.protocols import EvaluationContext
from spenn.evaluation.results import MetricScalar, SummaryResult


class SamplerStatsSummary:
    """Expose JSON-safe sampler statistics recorded by `MCMCGenerator`."""

    name = "sampler_stats"
    required_fields = frozenset({"generated"})

    def __init__(self, *, prefix: str = "sampler") -> None:
        self.prefix = str(prefix).strip("_")

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return scalar sampler stats from generated metadata."""

        stats = bundle.generated.metadata.get("sampler_stats", {})
        if not isinstance(stats, Mapping):
            raise TypeError("generated metadata field 'sampler_stats' must be a mapping")
        metrics: dict[str, MetricScalar] = {}
        for key, value in stats.items():
            if isinstance(value, bool):
                scalar: MetricScalar = value
            elif isinstance(value, int | float):
                scalar = value
            else:
                continue
            metric_key = str(key).strip()
            if not metric_key:
                continue
            metrics[f"{self.prefix}_{metric_key}" if self.prefix else metric_key] = scalar
        return SummaryResult(metrics=metrics)


__all__ = ["SamplerStatsSummary"]
