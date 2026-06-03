"""Statistical diagnostics for production sample streams."""

from __future__ import annotations

import math

import torch


def autocorrelation_by_lag(samples: torch.Tensor, max_lag: int | None = None) -> torch.Tensor:
    """Estimate normalized autocorrelation by lag.

    Parameters
    ----------
    samples : torch.Tensor
        Sample stream with shape ``[steps]`` or ``[steps, chains]``. The first
        axis must be the sequential sampling axis.
    max_lag : int or None, optional
        Maximum lag to evaluate. If ``None``, all lags up to ``steps - 1`` are
        evaluated.

    Returns
    -------
    torch.Tensor
        Mean normalized autocorrelation over non-constant chains with shape
        ``[max_lag + 1]``. Constant-chain estimates are returned as ``nan``.
    """

    values = _as_step_chain_tensor(samples)
    steps = values.shape[0]
    if steps < 2:
        return torch.full((0,), float("nan"), device=values.device, dtype=values.dtype)
    selected_max_lag = steps - 1 if max_lag is None else int(max_lag)
    if selected_max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    selected_max_lag = min(selected_max_lag, steps - 1)
    centered = values - values.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0)
    valid = variance > torch.finfo(values.dtype).eps
    if not bool(valid.any()):
        return torch.full((selected_max_lag + 1,), float("nan"), device=values.device, dtype=values.dtype)
    correlations = []
    for lag in range(selected_max_lag + 1):
        lhs = centered[: steps - lag]
        rhs = centered[lag:]
        covariance = (lhs * rhs).mean(dim=0)
        per_chain = covariance[valid] / variance[valid]
        correlations.append(per_chain.mean())
    return torch.stack(correlations)


def integrated_autocorrelation_time(samples: torch.Tensor, max_lag: int | None = None) -> float:
    """Estimate integrated autocorrelation time.

    Parameters
    ----------
    samples : torch.Tensor
        Sample stream with shape ``[steps]`` or ``[steps, chains]``. The first
        axis must be the sequential sampling axis.
    max_lag : int or None, optional
        Maximum lag to use in the positive-sequence estimate.

    Returns
    -------
    float
        Integrated autocorrelation time in sampling-step units. ``nan`` is
        returned when fewer than two steps or no non-constant chains are
        available.
    """

    correlation = autocorrelation_by_lag(samples, max_lag=max_lag)
    if correlation.numel() == 0 or not bool(torch.isfinite(correlation[0])):
        return float("nan")
    positive_terms = []
    for value in correlation[1:]:
        if not bool(torch.isfinite(value)) or float(value.item()) <= 0.0:
            break
        positive_terms.append(value)
    if not positive_terms:
        return 1.0
    tau = 1.0 + 2.0 * torch.stack(positive_terms).sum()
    return float(tau.item())


def effective_sample_size(samples: torch.Tensor, max_lag: int | None = None) -> float:
    """Estimate effective sample size from autocorrelation time.

    Parameters
    ----------
    samples : torch.Tensor
        Sample stream with shape ``[steps]`` or ``[steps, chains]``.
    max_lag : int or None, optional
        Maximum lag to use in the autocorrelation estimate.

    Returns
    -------
    float
        Effective sample size. ``nan`` is returned when autocorrelation time is
        unavailable.
    """

    values = _as_step_chain_tensor(samples)
    tau = integrated_autocorrelation_time(values, max_lag=max_lag)
    if not math.isfinite(tau) or tau <= 0.0:
        return float("nan")
    return float(values.numel() / tau)


def _as_step_chain_tensor(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim == 1:
        return samples.unsqueeze(-1)
    if samples.ndim == 2:
        return samples
    raise ValueError(f"samples must have shape [steps] or [steps, chains], got {tuple(samples.shape)}")
