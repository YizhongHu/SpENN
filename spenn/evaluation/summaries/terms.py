"""Hamiltonian term summaries."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from spenn.evaluation.bundle import EvaluationBundle
from spenn.evaluation.protocols import EvaluationContext
from spenn.evaluation.results import MetricScalar, SummaryResult


class HamiltonianTermSummary:
    """Summarize per-term local-energy components."""

    name = "hamiltonian_terms"
    required_fields = frozenset({"local_energy"})

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return mean/variance for each configured Hamiltonian term."""

        local = bundle.local_energy
        if local is None or local.term_energies is None:
            raise ValueError("HamiltonianTermSummary requires LocalEnergyValues.term_energies")
        metrics: dict[str, MetricScalar] = {}
        for name, values in local.term_energies.items():
            finite = values.detach().reshape(-1)
            finite = finite[torch.isfinite(finite)]
            if finite.numel() == 0:
                raise ValueError(f"term {name!r} has no finite samples")
            variance = finite.var(unbiased=False) if finite.numel() > 1 else torch.zeros((), dtype=finite.dtype, device=finite.device)
            metrics[f"term/{name}_mean"] = float(finite.mean().item())
            metrics[f"term/{name}_variance"] = float(variance.item())
        return SummaryResult(metrics=metrics)


__all__ = ["HamiltonianTermSummary"]
