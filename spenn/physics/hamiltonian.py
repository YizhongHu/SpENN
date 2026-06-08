"""Hamiltonian terms, local-energy results, and aggregation.

A Hamiltonian is represented simply as a list of `HamiltonianTerm`s. Each term
reports its contribution as a `LocalEnergyResult`, and the `local_energy` helper
evaluates every term and sums their contributions, optionally returning the
per-term decomposition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

import torch

from spenn.data.batch import ElectronBatch


@dataclass
class LocalEnergyResult:
    """Container for a decomposed local-energy evaluation.

    Parameters
    ----------
    total : torch.Tensor
        Summed local energy across all contributions, shape ``[batch]``.
    terms : dict[str, torch.Tensor]
        Per-term local energies keyed by ``HamiltonianTerm.name``.
    """

    total: torch.Tensor
    terms: dict[str, torch.Tensor] = field(default_factory=dict)


@runtime_checkable
class HamiltonianTerm(Protocol):
    """Protocol for a single Hamiltonian term.

    A term reports its contribution to the local energy as a
    `LocalEnergyResult` so that decompositions compose under summation.
    """

    name: str

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        """Return this term's local-energy contribution."""
        ...


def local_energy(
    terms: Sequence[HamiltonianTerm],
    wavefunction,
    batch: ElectronBatch,
    *,
    return_terms: bool = False,
) -> torch.Tensor | LocalEnergyResult:
    """Evaluate the local energy of a list of Hamiltonian terms.

    Parameters
    ----------
    terms : sequence of HamiltonianTerm
        Ordered Hamiltonian contributions to sum.
    wavefunction : callable
        Wavefunction model or exact reference returning ``WavefunctionOutput``.
    batch : ElectronBatch
        Electron configuration batch.
    return_terms : bool, optional
        If ``True``, return a ``LocalEnergyResult`` carrying the merged per-term
        decomposition; otherwise return the summed tensor directly.

    Returns
    -------
    torch.Tensor or LocalEnergyResult
        Summed local energy with shape ``[batch]``, or a decomposed result when
        ``return_terms=True``.
    """

    total: torch.Tensor | None = None
    merged: dict[str, torch.Tensor] = {}
    for term in terms:
        result = term.local_energy(wavefunction, batch)
        merged.update(result.terms)
        total = result.total if total is None else total + result.total
    if total is None:
        flat = batch.flatten_samples()
        total = torch.zeros(flat.batch_size, device=flat.device, dtype=flat.dtype)
    if return_terms:
        return LocalEnergyResult(total=total, terms=merged)
    return total


def _finite_stats(values: torch.Tensor) -> tuple[float, float]:
    """Return ``(mean_over_finite, nonfinite_fraction)`` for a tensor.

    The mean is ``nan`` when there are no finite entries, and the fraction is
    ``nan`` for an empty tensor. The finite mask is always checked before any
    mean is computed.
    """

    n = int(values.numel())
    finite_mask = torch.isfinite(values)
    n_finite = int(finite_mask.sum().item())
    mean = float(values[finite_mask].mean().item()) if n_finite > 0 else float("nan")
    nonfinite_fraction = float((n - n_finite) / n) if n > 0 else float("nan")
    return mean, nonfinite_fraction


def summarize_local_energy(
    result: LocalEnergyResult | torch.Tensor,
) -> dict[str, Any]:
    """Summarize a sampled local energy into scalar logging metrics.

    Handles all-finite, partially-nonfinite, all-nonfinite, and empty inputs,
    and per-term decompositions. All returned values are Python scalars
    suitable for CSV/JSONL logging. This summary is reference-free: comparison
    against a known energy is the job of `reference_energy_metrics` (typically
    driven by `spenn.callback.ReferenceEnergy`).

    Parameters
    ----------
    result : LocalEnergyResult or torch.Tensor
        Per-sample local energy, optionally with a per-term decomposition.

    Returns
    -------
    dict
        Scalar metrics. When no finite samples exist, ``energy_mean`` and
        ``energy_variance`` are ``nan`` and ``energy_stderr`` is ``inf``.
    """

    if isinstance(result, LocalEnergyResult):
        eloc, terms = result.total, result.terms
    else:
        eloc, terms = result, {}

    n = int(eloc.numel())
    finite_mask = torch.isfinite(eloc)
    n_finite = int(finite_mask.sum().item())
    if n_finite > 0:
        finite = eloc[finite_mask]
        mean = float(finite.mean().item())
        variance = float(finite.var(unbiased=False).item())
        stderr = float(finite.std(unbiased=False).item()) / math.sqrt(n_finite)
    else:
        mean = float("nan")
        variance = float("nan")
        stderr = float("inf")

    metrics: dict[str, Any] = {
        "n_samples": n,
        "n_finite_samples": n_finite,
        "nonfinite_energy_fraction": float((n - n_finite) / n) if n > 0 else float("nan"),
        "energy_mean": mean,
        "energy_stderr": stderr,
        "energy_variance": variance,
    }
    for name, value in terms.items():
        term_mean, term_nonfinite = _finite_stats(value)
        metrics[f"terms.{name}_mean"] = term_mean
        metrics[f"terms.{name}_nonfinite_fraction"] = term_nonfinite
    return metrics


def reference_energy_metrics(
    *,
    energy_mean: float,
    reference_energy: float,
) -> dict[str, float]:
    """Compare a mean energy against a known reference energy.

    Kept separate from `summarize_local_energy` so the trainer and energy
    summary never depend on benchmark reference values; reference comparison is
    an explicit run choice (see `spenn.callback.ReferenceEnergy`).

    Parameters
    ----------
    energy_mean : float
        Estimated mean energy.
    reference_energy : float
        Known reference energy to compare against.

    Returns
    -------
    dict
        ``reference_energy``, ``energy_error`` (mean minus reference), and
        ``abs_energy_error``.
    """

    error = float(energy_mean) - float(reference_energy)
    return {
        "reference_energy": float(reference_energy),
        "energy_error": error,
        "abs_energy_error": abs(error),
    }


__all__ = [
    "HamiltonianTerm",
    "LocalEnergyResult",
    "local_energy",
    "reference_energy_metrics",
    "summarize_local_energy",
]
