"""Hamiltonian terms, local-energy results, and aggregation.

A Hamiltonian is a collection of `HamiltonianTerm`s, given either as a sequence
or as a ``dict[str, HamiltonianTerm]`` that names each term explicitly. Dict
keys are the public, authoritative term names: they must be non-empty strings,
and the values must expose ``local_energy(wavefunction, batch)``. The
`local_energy` helper normalizes either form (see `normalize_hamiltonian_terms`),
evaluates every term, and sums their contributions, optionally returning the
per-term decomposition keyed by the resolved term names. Evaluation summaries
emit per-term metrics as ``terms.{name}_mean`` and
``terms.{name}_nonfinite_fraction``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from spenn.data.batch import ElectronBatch
from spenn.naming import camel_to_snake


@dataclass
class LocalEnergyResult:
    """Container for a decomposed local-energy evaluation.

    Parameters
    ----------
    total : torch.Tensor
        Summed local energy across all contributions, shape ``[batch]``.
    terms : dict[str, torch.Tensor]
        Per-term local energies keyed by the resolved term name. When produced
        by `local_energy`, names come from the ``dict`` key (named form) or the
        snake-case class name (sequence form), and are guaranteed unique.
    """

    total: torch.Tensor
    terms: dict[str, torch.Tensor] = field(default_factory=dict)


def normalize_hamiltonian_terms(
    terms: Mapping[Any, Any] | Sequence[Any],
) -> dict[str, "HamiltonianTerm"]:
    """Return an ordered ``{name: term}`` mapping from a dict or sequence.

    A ``dict[str, HamiltonianTerm]`` is used directly: its keys are the
    explicit, authoritative term names used in `LocalEnergyResult.terms` and
    downstream metrics. A sequence falls back to the snake-case class name of
    each term, suffixed with the term index when a class name repeats, so the
    resulting names are always unique.

    Names are enforced as non-empty strings and each value must expose a
    callable ``local_energy``; invalid specifications fail loudly here rather
    than later during evaluation.

    Parameters
    ----------
    terms : Mapping or Sequence of HamiltonianTerm
        Configured Hamiltonian terms, named (dict) or unnamed (sequence).

    Returns
    -------
    dict[str, HamiltonianTerm]
        Ordered mapping from resolved name to term.
    """

    if isinstance(terms, Mapping):
        normalized: dict[str, HamiltonianTerm] = {}
        for key, term in terms.items():
            if not isinstance(key, str):
                raise TypeError(f"hamiltonian term names must be strings, got {type(key).__name__}")
            _validate_hamiltonian_term(key, term)
            normalized[key] = term
        return normalized

    sequence = list(terms)
    base_names = [camel_to_snake(type(term).__name__) for term in sequence]
    counts = Counter(base_names)
    normalized = {}
    for index, (term, base) in enumerate(zip(sequence, base_names)):
        name = base if counts[base] == 1 else f"{base}_{index}"
        _validate_hamiltonian_term(name, term)
        normalized[name] = term
    return normalized


def _validate_hamiltonian_term(name: str, term: object) -> None:
    """Fail loudly on an empty term name or an invalid term specification."""

    if not name or not name.strip():
        raise ValueError("hamiltonian term names must be non-empty strings")
    if not callable(getattr(term, "local_energy", None)):
        raise TypeError(
            f"hamiltonian term {name!r} ({type(term).__name__}) must expose a callable "
            "local_energy(wavefunction, batch)"
        )


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
    terms: Mapping[str, HamiltonianTerm] | Sequence[HamiltonianTerm],
    wavefunction,
    batch: ElectronBatch,
    *,
    return_terms: bool = False,
) -> torch.Tensor | LocalEnergyResult:
    """Evaluate the local energy of a collection of Hamiltonian terms.

    Parameters
    ----------
    terms : Mapping or Sequence of HamiltonianTerm
        Hamiltonian contributions to sum. A ``dict[str, HamiltonianTerm]`` names
        terms by its non-empty string keys; a sequence falls back to snake-case
        class names (see `normalize_hamiltonian_terms`).
    wavefunction : callable
        Wavefunction model or exact reference returning ``WavefunctionOutput``.
    batch : ElectronBatch
        Electron configuration batch.
    return_terms : bool, optional
        If ``True``, return a ``LocalEnergyResult`` whose ``terms`` decomposition
        is keyed by the resolved (unique) term names; otherwise return the summed
        tensor directly.

    Returns
    -------
    torch.Tensor or LocalEnergyResult
        Summed local energy with shape ``[batch]``, or a decomposed result when
        ``return_terms=True``.
    """

    normalized = normalize_hamiltonian_terms(terms)
    total: torch.Tensor | None = None
    decomposition: dict[str, torch.Tensor] = {}
    for name, term in normalized.items():
        result = term.local_energy(wavefunction, batch)
        decomposition[name] = result.total
        total = result.total if total is None else total + result.total
    if total is None:
        flat = batch.flatten_samples()
        total = torch.zeros(flat.batch_size, device=flat.device, dtype=flat.dtype)
    if return_terms:
        return LocalEnergyResult(total=total, terms=decomposition)
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
    and per-term decompositions. Per-term metrics are named
    ``terms.{name}_mean`` and ``terms.{name}_nonfinite_fraction``, where
    ``name`` is the resolved Hamiltonian term name. All returned values are
    Python scalars suitable for CSV/JSONL logging. This summary is
    reference-free: comparison against a known energy is the job of
    `reference_energy_metrics` (typically driven by
    `spenn.callback.ReferenceEnergy`).

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

    .. deprecated::
        Transitional. Reference/exact-energy comparison belongs to the
        evaluation-side diagnostics introduced in PR6 (``EnergyEvaluation``),
        not to this physics module. This helper is kept only so the Hooke eval
        smoke path keeps working until then, and is scheduled for removal when
        PR6 diagnostics land. ``VMCTrainer`` must never call it: training
        metrics stay VMC-native, and reference comparison stays in evaluation.

    Kept separate from `summarize_local_energy` so the trainer and energy
    summary never depend on benchmark reference values; reference comparison is
    an explicit run/diagnostics choice.

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
    "normalize_hamiltonian_terms",
    "reference_energy_metrics",
    "summarize_local_energy",
]
