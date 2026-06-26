"""Canonical VMC training objective and VMC-native training metrics.

This module is the single source of truth for the VMC score-function objective
used by `VMCTrainer`. It returns one differentiable scalar ``loss`` for
``optimizer.step()`` alongside detached, JSON-safe training metrics. Per-term
local-energy summaries are metrics (not loss components): they may be computed
from the same local-energy batch, but they never form a second public objective
surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from spenn.dependencies import require_torch

torch = require_torch(feature="VMC objective")


@dataclass(frozen=True)
class VMCObjectiveResult:
    """Differentiable VMC objective plus detached JSON-safe training metrics.

    Parameters
    ----------
    loss : torch.Tensor
        Differentiable scalar surrogate objective for ``optimizer.step()``. The
        only object in this result that carries an autograd graph.
    metrics : dict
        Detached, JSON-safe training metrics (Python scalars only).
    """

    loss: torch.Tensor
    metrics: dict[str, float | int]


def compute_vmc_objective(
    logabs: torch.Tensor,
    local_energy: torch.Tensor,
    *,
    scale_factor: float = 2.0,
) -> VMCObjectiveResult:
    """Compute the VMC score-function objective and training metrics.

    The returned loss is differentiable with respect to ``logabs``. Local-energy
    values are detached before forming the score-function objective, so the
    gradient flows only through ``logabs``.

    Non-finite local-energy samples are excluded from the objective and from the
    energy summary metrics. The function raises if no finite samples remain.

    Parameters
    ----------
    logabs : torch.Tensor
        Log absolute wavefunction values with shape ``[batch]``. Carries the
        autograd graph used for backpropagation.
    local_energy : torch.Tensor
        Per-sample total local energy with shape ``[batch]``.
    scale_factor : float, optional
        Multiplicative factor on the score-function objective. The default ``2``
        corresponds to gradients of an expectation under ``|psi|^2``.

    Returns
    -------
    VMCObjectiveResult
        Differentiable ``loss`` and detached, JSON-safe ``metrics``.

    Raises
    ------
    ValueError
        If ``logabs`` and ``local_energy`` shapes differ, or if no finite
        local-energy sample remains.
    """

    if logabs.shape != local_energy.shape:
        raise ValueError(
            "logabs and local_energy must have the same shape, "
            f"got {tuple(logabs.shape)} and {tuple(local_energy.shape)}"
        )

    finite_mask = torch.isfinite(local_energy)
    n_total = int(local_energy.numel())
    n_finite = int(finite_mask.sum().item())

    if n_finite == 0:
        raise ValueError("cannot compute VMC objective: no finite local-energy samples")

    finite_logabs = logabs[finite_mask]
    finite_energy = local_energy[finite_mask].detach()

    energy = finite_energy.mean()
    centered_energy = finite_energy - energy

    loss = scale_factor * torch.mean(centered_energy * finite_logabs)

    if n_finite > 1:
        energy_variance = finite_energy.var(unbiased=False)
    else:
        energy_variance = torch.zeros((), device=finite_energy.device, dtype=finite_energy.dtype)

    energy_std = torch.sqrt(energy_variance)
    energy_stderr = energy_std / float(n_finite) ** 0.5

    metrics: dict[str, float | int] = {
        "loss": float(loss.detach().item()),
        "energy": float(energy.detach().item()),
        "energy_variance": float(energy_variance.detach().item()),
        "energy_std": float(energy_std.detach().item()),
        "energy_stderr": float(energy_stderr.detach().item()),
        "local_energy_n_finite": n_finite,
        "local_energy_n_total": n_total,
        "local_energy_finite_fraction": float(n_finite / n_total) if n_total else 0.0,
        "local_energy_nonfinite_count": n_total - n_finite,
    }

    return VMCObjectiveResult(loss=loss, metrics=metrics)


def hamiltonian_term_metric_prefix(name: str) -> str:
    """Return the metric-key prefix for a named Hamiltonian term.

    The prefix is derived from the resolved term name (the ``dict`` key, or the
    snake-case class name for a sequence; see
    `spenn.physics.hamiltonian.normalize_hamiltonian_terms`). Names are unique,
    so prefixes are deterministic and collision-free. Training per-term metrics
    use this prefix directly for the finite mean and append suffixes such as
    ``_variance``, ``_std``, ``_stderr``, ``_n_finite``, ``_n_total``,
    ``_finite_fraction``, and ``_nonfinite_count`` for companion statistics.
    """

    return f"energy_term_{name}"


def summarize_local_energy_terms(
    terms: Mapping[str, torch.Tensor],
) -> dict[str, float | int]:
    """Summarize per-Hamiltonian-term local-energy tensors as training metrics.

    Term metric keys are derived from the resolved term names (see
    `hamiltonian_term_metric_prefix`). For a resolved name ``kinetic``, the
    finite mean is logged as ``energy_term_kinetic`` and companion statistics
    use suffixes like ``energy_term_kinetic_variance``. These are metrics only
    -- they never form part of the optimizer objective.

    Parameters
    ----------
    terms : Mapping of str to torch.Tensor
        Per-term local-energy tensors keyed by resolved term name, as produced
        by ``local_energy(..., return_terms=True).terms``.

    Returns
    -------
    dict
        Detached, JSON-safe per-term metrics (Python scalars only).

    Raises
    ------
    ValueError
        If any term has no finite samples.
    """

    metrics: dict[str, float | int] = {}

    for name, values in terms.items():
        prefix = hamiltonian_term_metric_prefix(name)

        finite_mask = torch.isfinite(values)
        n_total = int(values.numel())
        n_finite = int(finite_mask.sum().item())

        if n_finite == 0:
            raise ValueError(f"cannot summarize local-energy term {prefix}: no finite samples")

        finite_values = values[finite_mask].detach()
        energy = finite_values.mean()

        if n_finite > 1:
            variance = finite_values.var(unbiased=False)
        else:
            variance = torch.zeros((), device=finite_values.device, dtype=finite_values.dtype)

        std = torch.sqrt(variance)
        stderr = std / float(n_finite) ** 0.5

        metrics[prefix] = float(energy.detach().item())
        metrics[f"{prefix}_variance"] = float(variance.detach().item())
        metrics[f"{prefix}_std"] = float(std.detach().item())
        metrics[f"{prefix}_stderr"] = float(stderr.detach().item())
        metrics[f"{prefix}_n_finite"] = n_finite
        metrics[f"{prefix}_n_total"] = n_total
        metrics[f"{prefix}_finite_fraction"] = float(n_finite / n_total) if n_total else 0.0
        metrics[f"{prefix}_nonfinite_count"] = n_total - n_finite

    return metrics


def summarize_logabs(logabs: torch.Tensor) -> dict[str, float]:
    """Summarize log-amplitude values into finite-aware scalar metrics.

    Parameters
    ----------
    logabs : torch.Tensor
        Log absolute wavefunction values with shape ``[batch]``.

    Returns
    -------
    dict
        Scalar metrics ``logabs_mean``, ``logabs_min``, ``logabs_max``, and
        ``nonfinite_logabs_fraction``. Statistics are computed over finite
        entries and are ``nan`` when no entry is finite.
    """

    n = int(logabs.numel())
    finite_mask = torch.isfinite(logabs)
    n_finite = int(finite_mask.sum().item())
    if n_finite > 0:
        finite = logabs[finite_mask]
        mean = float(finite.mean().item())
        minimum = float(finite.min().item())
        maximum = float(finite.max().item())
    else:
        mean = float("nan")
        minimum = float("nan")
        maximum = float("nan")
    return {
        "logabs_mean": mean,
        "logabs_min": minimum,
        "logabs_max": maximum,
        "nonfinite_logabs_fraction": float((n - n_finite) / n) if n > 0 else float("nan"),
    }


__all__ = [
    "VMCObjectiveResult",
    "compute_vmc_objective",
    "hamiltonian_term_metric_prefix",
    "summarize_local_energy_terms",
    "summarize_logabs",
]
