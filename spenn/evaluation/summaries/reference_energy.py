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

    def __init__(self, *, reference_energy: float, allow_phase: str = "eval") -> None:
        self.reference_energy = float(reference_energy)
        self.allow_phase = str(allow_phase)

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return reference-energy comparison metrics."""

        if context.phase != self.allow_phase:
            raise ValueError(
                f"ReferenceEnergySummary is only allowed in phase {self.allow_phase!r}; "
                f"got {context.phase!r}"
            )
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
