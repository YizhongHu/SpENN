"""Local-energy scalar summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from spenn.evaluation.bundle import EvaluationBundle
from spenn.evaluation.protocols import EvaluationContext
from spenn.evaluation.results import MetricScalar, SummaryResult


class LocalEnergySummary:
    """Summarize finite local-energy samples."""

    name = "local_energy"
    required_fields = frozenset({"local_energy"})

    def __init__(
        self,
        *,
        quantiles: Sequence[float] = (0.01, 0.05, 0.5, 0.95, 0.99),
        prefix: str = "local_energy",
    ) -> None:
        self.quantiles = tuple(_validate_quantile(q) for q in quantiles)
        self.prefix = str(prefix)

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return finite-value moments, quantiles, and nonfinite counts."""

        values = bundle.local_energy
        if values is None:
            raise ValueError("LocalEnergySummary requires bundle.local_energy")
        metrics = summarize_values(values.local_energy, quantiles=self.quantiles, prefix=self.prefix)
        return SummaryResult(metrics=metrics)


def summarize_values(
    values: torch.Tensor,
    *,
    quantiles: Sequence[float],
    prefix: str,
) -> dict[str, MetricScalar]:
    """Summarize finite tensor values with a metric prefix."""

    flat = values.detach().reshape(-1)
    finite_mask = torch.isfinite(flat)
    n_total = int(flat.numel())
    n_finite = int(finite_mask.sum().item())
    if n_finite == 0:
        raise ValueError(f"cannot summarize {prefix}: no finite samples")
    finite = flat[finite_mask]
    variance = finite.var(unbiased=False) if n_finite > 1 else torch.zeros((), dtype=finite.dtype, device=finite.device)
    std = torch.sqrt(variance)
    stderr = std / float(n_finite) ** 0.5
    metrics: dict[str, MetricScalar] = {
        f"{prefix}_mean": float(finite.mean().item()),
        f"{prefix}_variance": float(variance.item()),
        f"{prefix}_std": float(std.item()),
        f"{prefix}_stderr": float(stderr.item()),
        f"{prefix}_min": float(finite.min().item()),
        f"{prefix}_max": float(finite.max().item()),
        f"{prefix}_finite_fraction": float(n_finite / n_total) if n_total else 0.0,
        f"{prefix}_nonfinite_count": n_total - n_finite,
        f"{prefix}_n_finite": n_finite,
        f"{prefix}_n_total": n_total,
    }
    metrics.update(_quantile_metrics(prefix, finite, quantiles))
    return metrics


def _quantile_metrics(prefix: str, values: torch.Tensor, quantiles: Sequence[float]) -> dict[str, float]:
    if not quantiles:
        return {}
    q_tensor = torch.tensor(tuple(quantiles), device=values.device, dtype=values.dtype)
    q_values = torch.quantile(values, q_tensor)
    return {
        f"{prefix}_{_quantile_label(q)}": float(value.item())
        for q, value in zip(quantiles, q_values, strict=True)
    }


def _validate_quantile(value: float) -> float:
    q = float(value)
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantiles must lie in [0, 1], got {q}")
    return q


def _quantile_label(value: float) -> str:
    known = {
        0.001: "q001",
        0.01: "q01",
        0.05: "q05",
        0.5: "q50",
        0.95: "q95",
        0.99: "q99",
        0.999: "q999",
    }
    for key, label in known.items():
        if math.isclose(value, key, rel_tol=0.0, abs_tol=1.0e-12):
            return label
    return "q" + f"{100.0 * value:g}".replace(".", "p")


__all__ = ["LocalEnergySummary", "summarize_values"]
