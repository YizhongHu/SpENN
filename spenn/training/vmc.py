"""VMC surrogate loss and log-amplitude summaries for the training loop."""

from __future__ import annotations

import torch


def vmc_surrogate_loss(logabs: torch.Tensor, local_energy_values: torch.Tensor) -> torch.Tensor:
    """Return the score-function VMC surrogate loss.

    Implements the centered score-function objective whose gradient estimates
    the gradient of the energy expectation under ``|psi|^2``. The local energy
    is detached, so gradients flow only through ``logabs``. Samples with a
    non-finite local energy are masked out.

    Parameters
    ----------
    logabs : torch.Tensor
        Log absolute wavefunction values with shape ``[batch]``. Carries the
        autograd graph used for backpropagation.
    local_energy_values : torch.Tensor
        Per-sample local energy with shape ``[batch]``.

    Returns
    -------
    torch.Tensor
        Scalar surrogate loss.

    Raises
    ------
    RuntimeError
        If no sample has a finite local energy.
    """

    finite = torch.isfinite(local_energy_values)
    if not bool(finite.any()):
        raise RuntimeError("vmc_surrogate_loss received no finite local-energy samples")
    energy = local_energy_values[finite].detach()
    weighted_logabs = logabs[finite]
    return 2.0 * ((energy - energy.mean()) * weighted_logabs).mean()


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


__all__ = ["summarize_logabs", "vmc_surrogate_loss"]
