"""Metadata summaries for evaluation generators."""

from __future__ import annotations

import json
from pathlib import Path

from tpen.evaluation.bundle import EvaluationBundle
from tpen.evaluation.generators.trajectory import SAMPLER_TRAJECTORY_DIAGNOSTICS_KEY
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.results import ArtifactRecord, MetricScalar, SummaryResult
from tpen.sampling.stats import SamplerStats
from tpen.sampling.trajectory import SamplerTrajectoryDiagnostics

SAMPLER_TRAJECTORY_DIAGNOSTICS_FILENAME = "sampler_trajectory_diagnostics.json"
"""Versioned draw-resolved sampler-health sidecar filename."""


class SamplerStatsSummary:
    """Expose JSON-safe sampler statistics recorded by `MCMCGenerator`."""

    name = "sampler_stats"
    required_fields = frozenset({"generated"})

    def __init__(
        self,
        *,
        prefix: str = "sampler",
        trajectory_diagnostics_filename: str = SAMPLER_TRAJECTORY_DIAGNOSTICS_FILENAME,
    ) -> None:
        self.prefix = str(prefix).strip("_")
        self.trajectory_diagnostics_filename = str(trajectory_diagnostics_filename).strip()
        if not self.trajectory_diagnostics_filename:
            raise ValueError("trajectory_diagnostics_filename must be non-empty")

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
        trajectory_diagnostics = bundle.generated.metadata.get(
            SAMPLER_TRAJECTORY_DIAGNOSTICS_KEY
        )
        artifacts: tuple[ArtifactRecord, ...] = ()
        if trajectory_diagnostics is not None:
            if not isinstance(trajectory_diagnostics, SamplerTrajectoryDiagnostics):
                raise TypeError(
                    f"generated metadata field {SAMPLER_TRAJECTORY_DIAGNOSTICS_KEY!r} "
                    "must be a SamplerTrajectoryDiagnostics record"
                )
            for key, value in trajectory_diagnostics.as_metrics().items():
                metrics[_prefixed(self.prefix, key)] = value
            path = Path(context.task_output_dir) / self.trajectory_diagnostics_filename
            path.parent.mkdir(parents=True, exist_ok=True)
            # Task output directories are attempt-local. Refuse to replace a
            # prior sidecar, because that would erase the draw series whose
            # scalar projection is being logged.
            with path.open("x", encoding="utf-8") as handle:
                json.dump(trajectory_diagnostics.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
            artifacts = (
                ArtifactRecord(
                    name="sampler_trajectory_diagnostics",
                    kind="sampler_trajectory_diagnostics",
                    path=path,
                    metadata={
                        "schema": "sampler_trajectory_diagnostics/v1",
                        "retained_draw_count": len(trajectory_diagnostics.retained_draws),
                        "discarded_draw_count": len(trajectory_diagnostics.discarded_draws),
                        "draw_stride": trajectory_diagnostics.draw_stride,
                        "intermediate_sampler_steps_observed": False,
                    },
                ),
            )
        return SummaryResult(metrics=metrics, artifacts=artifacts)


def _prefixed(prefix: str, key: str) -> str:
    """Return one summary metric key under its configured sampler prefix."""

    return f"{prefix}_{key}" if prefix else key


__all__ = ["SAMPLER_TRAJECTORY_DIAGNOSTICS_FILENAME", "SamplerStatsSummary"]
