"""Energy diagnostics for sampled evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spenn.diagnostics.base import EvaluationContext, JsonScalar
from spenn.training.vmc import summarize_local_energy_terms


@dataclass(frozen=True)
class EnergyEvaluation:
    """Summarize total and optional per-term local energy for evaluation.

    Parameters
    ----------
    name : str, optional
        Diagnostic name. PR6 emits canonical flat energy metric keys, so this
        name is metadata for the runner and for future collision policy.
    reference_energy : float or None, optional
        Optional exact/reference energy. When provided, signed and absolute
        energy errors are emitted by this evaluation diagnostic.
    include_terms : bool, optional
        Whether to summarize ``EvaluationContext.local_energy_terms``. The
        context must include term energies when this is ``True``.
    """

    name: str = "energy"
    reference_energy: float | None = None
    include_terms: bool = False

    def evaluate(self, context: EvaluationContext) -> dict[str, JsonScalar]:
        """Return flat JSON-safe energy metrics."""

        metrics = _summarize_total_energy(context.local_energy)
        if self.reference_energy is not None:
            error = float(metrics["energy"]) - float(self.reference_energy)
            metrics["energy_error"] = error
            metrics["energy_abs_error"] = abs(error)

        if self.include_terms:
            if context.local_energy_terms is None:
                raise ValueError(
                    "EnergyEvaluation(include_terms=True) requires "
                    "EvaluationContext.local_energy_terms; set Evaluate(return_terms=True)."
                )
            metrics.update(summarize_local_energy_terms(context.local_energy_terms))

        return metrics


def _summarize_total_energy(local_energy: torch.Tensor) -> dict[str, JsonScalar]:
    """Summarize finite local-energy samples with VMC-compatible metric names."""

    finite_mask = torch.isfinite(local_energy)
    n_total = int(local_energy.numel())
    n_finite = int(finite_mask.sum().item())

    if n_finite == 0:
        raise ValueError("cannot summarize evaluation energy: no finite local-energy samples")

    finite_energy = local_energy[finite_mask].detach()
    energy = finite_energy.mean()

    if n_finite > 1:
        variance = finite_energy.var(unbiased=False)
    else:
        variance = torch.zeros((), device=finite_energy.device, dtype=finite_energy.dtype)

    std = torch.sqrt(variance)
    stderr = std / float(n_finite) ** 0.5

    return {
        "energy": float(energy.item()),
        "energy_variance": float(variance.item()),
        "energy_std": float(std.item()),
        "energy_stderr": float(stderr.item()),
        "local_energy_n_finite": n_finite,
        "local_energy_n_total": n_total,
        "local_energy_finite_fraction": float(n_finite / n_total) if n_total else 0.0,
        "local_energy_nonfinite_count": n_total - n_finite,
    }


__all__ = ["EnergyEvaluation"]
