"""Hamiltonian terms, local-energy results, and aggregation.

A Hamiltonian is a collection of `HamiltonianTerm`s, given either as a sequence
or as a ``dict[str, HamiltonianTerm]`` that names each term explicitly. Dict
keys are the public, authoritative term names: they must be non-empty strings,
and the values must expose ``local_energy(wavefunction, batch)``. The
`local_energy` helper normalizes either form (see `normalize_hamiltonian_terms`),
evaluates every term, and sums their contributions, optionally returning the
per-term decomposition keyed by the resolved term names. Evaluation summaries
use canonical flat metric keys such as ``energy`` and
``energy_term_{name}``; hierarchy belongs in the logging namespace.
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
) -> dict[str, Any]:
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
    terms: Mapping[Any, Any] | Sequence[Any],
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
    batch_size = batch.flatten_samples().batch_size
    total: torch.Tensor | None = None
    decomposition: dict[str, torch.Tensor] = {}
    for name, term in normalized.items():
        result = term.local_energy(wavefunction, batch)
        result = _validate_local_energy_result(name, result, batch_size=batch_size)
        decomposition[name] = result.total
        total = result.total if total is None else total + result.total
    if total is None:
        flat = batch.flatten_samples()
        total = torch.zeros(flat.batch_size, device=flat.device, dtype=flat.dtype)
    if return_terms:
        return LocalEnergyResult(total=total, terms=decomposition)
    return total


def _validate_local_energy_result(
    name: str,
    result: object,
    *,
    batch_size: int,
) -> LocalEnergyResult:
    """Validate the object returned by one Hamiltonian term."""

    if not isinstance(result, LocalEnergyResult):
        raise TypeError(
            f"hamiltonian term {name!r} must return LocalEnergyResult, got {type(result).__name__}"
        )
    if not isinstance(result.total, torch.Tensor):
        raise TypeError(f"hamiltonian term {name!r} total must be a torch.Tensor")
    expected_shape = (batch_size,)
    if tuple(result.total.shape) != expected_shape:
        raise ValueError(
            f"hamiltonian term {name!r} total must have shape {expected_shape}, "
            f"got {tuple(result.total.shape)}"
        )
    if not isinstance(result.terms, Mapping):
        raise TypeError(f"hamiltonian term {name!r} terms must be a mapping")
    for term_name, value in result.terms.items():
        if not isinstance(term_name, str) or not term_name.strip():
            raise ValueError(f"hamiltonian term {name!r} returned an empty decomposition name")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"hamiltonian term {name!r} decomposition {term_name!r} must be a torch.Tensor")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"hamiltonian term {name!r} decomposition {term_name!r} must have shape "
                f"{expected_shape}, got {tuple(value.shape)}"
            )
    return result


def summarize_local_energy(
    result: LocalEnergyResult | torch.Tensor,
) -> dict[str, Any]:
    """Summarize a sampled local energy into scalar logging metrics.

    Handles all-finite, partially-nonfinite, all-nonfinite, and empty inputs,
    and per-term decompositions. Returned keys follow the metric naming
    convention: callers provide the logging namespace, while this helper emits
    flat leaf keys such as ``energy``, ``local_energy_n_finite``, and
    ``energy_term_{name}``. This summary is reference-free; benchmark
    comparison belongs to evaluation diagnostics.

    Parameters
    ----------
    result : LocalEnergyResult or torch.Tensor
        Per-sample local energy, optionally with a per-term decomposition.

    Returns
    -------
    dict
        Scalar metrics. When no finite samples exist, ``energy`` and
        ``energy_variance`` are ``nan`` and ``energy_stderr`` is ``inf``.
    """

    if isinstance(result, LocalEnergyResult):
        eloc, terms = result.total, result.terms
    else:
        eloc, terms = result, {}

    metrics = _summarize_values("", eloc)
    for name, value in terms.items():
        metrics.update(_summarize_values(f"energy_term_{name}", value))
    return metrics


def _summarize_values(prefix: str, values: torch.Tensor) -> dict[str, Any]:
    """Return canonical finite-aware energy metrics for one value tensor."""

    n_total = int(values.numel())
    finite_mask = torch.isfinite(values)
    n_finite = int(finite_mask.sum().item())
    if n_finite > 0:
        finite = values[finite_mask]
        energy = float(finite.mean().item())
        variance = float(finite.var(unbiased=False).item()) if n_finite > 1 else 0.0
        std = math.sqrt(variance)
        stderr = std / math.sqrt(n_finite)
    else:
        energy = float("nan")
        variance = float("nan")
        std = float("nan")
        stderr = float("inf")

    if prefix:
        return {
            prefix: energy,
            f"{prefix}_variance": variance,
            f"{prefix}_std": std,
            f"{prefix}_stderr": stderr,
            f"{prefix}_n_finite": n_finite,
            f"{prefix}_n_total": n_total,
            f"{prefix}_finite_fraction": float(n_finite / n_total) if n_total else 0.0,
            f"{prefix}_nonfinite_count": n_total - n_finite,
        }

    return {
        "energy": energy,
        "energy_variance": variance,
        "energy_std": std,
        "energy_stderr": stderr,
        "local_energy_n_finite": n_finite,
        "local_energy_n_total": n_total,
        "local_energy_finite_fraction": float(n_finite / n_total) if n_total else 0.0,
        "local_energy_nonfinite_count": n_total - n_finite,
    }


def reference_energy_metrics(
    *,
    energy: float,
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
    energy : float
        Estimated mean energy.
    reference_energy : float
        Known reference energy to compare against.

    Returns
    -------
    dict
        ``reference_energy``, ``energy_error`` (mean minus reference), and
        ``energy_abs_error``.
    """

    error = float(energy) - float(reference_energy)
    return {
        "reference_energy": float(reference_energy),
        "energy_error": error,
        "energy_abs_error": abs(error),
    }


__all__ = [
    "HamiltonianTerm",
    "LocalEnergyResult",
    "local_energy",
    "normalize_hamiltonian_terms",
    "reference_energy_metrics",
    "summarize_local_energy",
]
