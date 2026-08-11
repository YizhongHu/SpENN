"""Metadata summaries for evaluation generators."""

from __future__ import annotations

from tpen.evaluation.bundle import EvaluationBundle
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.results import MetricScalar, SummaryResult
from tpen.sampling.stats import SamplerStats


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

        stats = bundle.generated.metadata.get("sampler_stats")
        if stats is None:
            return SummaryResult(metrics={})
        if not isinstance(stats, SamplerStats):
            raise TypeError(
                "generated metadata field 'sampler_stats' must be a SamplerStats record"
            )
        metrics: dict[str, MetricScalar] = {}
        # The record composes the metric names; this summary only prefixes them.
        for key, value in stats.as_metrics().items():
            if isinstance(value, bool):
                scalar: MetricScalar = value
            elif isinstance(value, int | float):
                scalar = value
            else:
                continue
            metric_key = key.strip()
            if not metric_key:
                continue
            metrics[f"{self.prefix}_{metric_key}" if self.prefix else metric_key] = scalar
        return SummaryResult(metrics=metrics)


__all__ = ["SamplerStatsSummary"]
