"""Hamiltonian terms, local-energy results, and aggregation.

A Hamiltonian is represented simply as a list of `HamiltonianTerm`s. Each term
reports its contribution as a `LocalEnergyResult`, and the `local_energy` helper
evaluates every term and sums their contributions, optionally returning the
per-term decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

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


__all__ = ["HamiltonianTerm", "LocalEnergyResult", "local_energy"]
