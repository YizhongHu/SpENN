"""Hamiltonian terms, local-energy results, and typed evaluation.

A Hamiltonian is a collection of `HamiltonianTerm`s, given either as a sequence
or as a ``dict[str, HamiltonianTerm]`` that names each term explicitly. Dict
keys are the public, authoritative term names: they must be non-empty strings,
and the values must expose ``local_energy(wavefunction, batch)``.

Evaluation goes through a typed evaluator seam: `NaiveLocalEnergyEvaluator`
consumes a `NaiveLocalEnergyContext` and owns term normalization (see
`normalize_hamiltonian_terms`), ordered per-term evaluation, validation, and
summation, optionally returning the per-term decomposition keyed by the
resolved term names. The `local_energy` helper is an explicit delegate to the
naive evaluator and remains the single default path; the naive evaluator is
the permanent numerical reference for any future optimized evaluator behind
the `LocalEnergyEvaluator` protocol. Evaluation summaries use canonical flat
metric keys such as ``energy`` and ``energy_term_{name}``; hierarchy belongs
in the logging namespace.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import torch

from tpen.data.batch import ElectronBatch
from tpen.naming import camel_to_snake


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


ContextT = TypeVar("ContextT")


@dataclass(frozen=True)
class NaiveLocalEnergyContext:
    """Typed input context for the naive local-energy evaluator.

    Parameters
    ----------
    wavefunction : callable
        Wavefunction model or exact reference returning ``WavefunctionOutput``.
        The narrowest existing callable contract is reused deliberately; no
        ansatz-specific base class exists solely for this interface.
    batch : ElectronBatch
        Electron configuration batch.
    """

    wavefunction: object
    batch: ElectronBatch


class LocalEnergyEvaluator(Protocol, Generic[ContextT]):
    """Protocol for local-energy evaluators over an open set of terms.

    Evaluators consume an explicit typed context, never an arbitrary mapping.
    Each evaluator declares the narrow context type it requires; the naive
    evaluator below is the permanent numerical reference for every future
    optimized evaluator.
    """

    def evaluate(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        context: ContextT,
        *,
        return_terms: bool = False,
    ) -> torch.Tensor | LocalEnergyResult:
        """Evaluate the local energy for one typed context."""
        ...


class NaiveLocalEnergyEvaluator(LocalEnergyEvaluator[NaiveLocalEnergyContext]):
    """Reference evaluator: ordered per-term autodiff evaluation and summation.

    Owns the aggregation loop previously inlined in `local_energy`: term
    normalization, per-term evaluation through the open ``HamiltonianTerm``
    protocol, result validation, named decomposition, and summation — with
    unchanged arithmetic and ordering. This path remains the independent
    numerical reference for any later optimized evaluator.
    """

    def evaluate(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        context: NaiveLocalEnergyContext,
        *,
        return_terms: bool = False,
    ) -> torch.Tensor | LocalEnergyResult:
        """Evaluate the summed (optionally decomposed) local energy."""

        if not isinstance(context, NaiveLocalEnergyContext):
            raise TypeError(
                "NaiveLocalEnergyEvaluator requires a NaiveLocalEnergyContext, "
                f"got {type(context).__name__}"
            )
        normalized = normalize_hamiltonian_terms(terms)
        batch = context.batch
        batch_size = batch.flatten_samples().batch_size
        total: torch.Tensor | None = None
        decomposition: dict[str, torch.Tensor] = {}
        for name, term in normalized.items():
            result = term.local_energy(context.wavefunction, batch)
            result = _validate_local_energy_result(name, result, batch_size=batch_size)
            decomposition[name] = result.total
            total = result.total if total is None else total + result.total
        if total is None:
            flat = batch.flatten_samples()
            total = torch.zeros(flat.batch_size, device=flat.device, dtype=flat.dtype)
        if return_terms:
            return LocalEnergyResult(total=total, terms=decomposition)
        return total


_NAIVE_EVALUATOR = NaiveLocalEnergyEvaluator()


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

    # Explicit delegate: this entry point stays the single default path used
    # by training and evaluation; the naive evaluator owns the aggregation.
    context = NaiveLocalEnergyContext(wavefunction=wavefunction, batch=batch)
    return _NAIVE_EVALUATOR.evaluate(terms, context, return_terms=return_terms)


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

    Notes
    -----
    ``energy_stderr`` and every ``{prefix}_stderr`` here is an **IID-only**
    standard error, ``sigma / sqrt(N)``, identical in meaning to the metric of
    the same name from `tpen.training.vmc.compute_vmc_objective`. It ignores serial
    correlation between MCMC walkers and so understates the true uncertainty by
    roughly ``sqrt(tau_int)``.

    The correlation-aware quantity is the MCSE from
    :func:`tpen.statistics.produce_trajectory_statistics`, which needs a
    ``[draw, walker]`` trajectory rather than the flat sample tensor available
    here. Do not reinterpret this metric as an MCSE.
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
    """Return canonical finite-aware energy metrics for one value tensor.

    ``{prefix}_stderr`` is an **IID-only** standard error; see the Notes on
    `summarize_local_energy`. It is never an MCSE.
    """

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


__all__ = [
    "HamiltonianTerm",
    "LocalEnergyEvaluator",
    "LocalEnergyResult",
    "NaiveLocalEnergyContext",
    "NaiveLocalEnergyEvaluator",
    "local_energy",
    "normalize_hamiltonian_terms",
    "summarize_local_energy",
]
