"""Reference-energy summaries for final evaluation."""

from __future__ import annotations

import torch

from spenn.evaluation.bundle import EvaluationBundle
from spenn.evaluation.protocols import EvaluationContext
from spenn.evaluation.results import SummaryResult


class ReferenceEnergySummary:
    """Compare sampled energy against an eval-only reference energy."""

    name = "reference_energy"
    required_fields = frozenset({"local_energy"})

    def __init__(self, *, reference_energy: float) -> None:
        self.reference_energy = float(reference_energy)

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return reference-energy comparison metrics."""

        local = bundle.local_energy
        if local is None:
            raise ValueError("ReferenceEnergySummary requires bundle.local_energy")
        values = local.local_energy.detach().reshape(-1)
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            raise ValueError("cannot compare reference energy: no finite local-energy samples")
        energy = float(finite.mean().item())
        error = energy - self.reference_energy
        return SummaryResult(
            metrics={
                "reference_energy": self.reference_energy,
                "energy_error": error,
                "energy_abs_error": abs(error),
            }
        )


__all__ = ["ReferenceEnergySummary"]
