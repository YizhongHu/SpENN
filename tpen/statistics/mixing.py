"""Split-chain mixing diagnostics, kept separate from effective sample size.

ESS says how much independent information a trajectory carries *if* the chains
are sampling the intended distribution. It says nothing about whether they are.
A short chain stuck in one mode can report a comfortable ESS and a completely
wrong mean, so mixing is estimated here, reported alongside ESS, and never
folded into it.

The statistic is the standard split-Rhat: every chain is halved before the
between- and within-chain variances are compared, so a single chain that drifts
across its own length is caught even when several chains agree with each other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = ["MixingDiagnostics", "split_r_hat"]


@dataclass(frozen=True)
class MixingDiagnostics:
    """Split-Rhat outcome for one observable.

    Attributes
    ----------
    r_hat : float or None
        Split-Rhat, or ``None`` when it could not be computed. Values near
        ``1.0`` are consistent with mixing; larger values mean the split halves
        disagree by more than their internal spread.
    n_split_chains : int
        Number of half-chains compared, i.e. twice the walker count.
    draws_per_split_chain : int
        Draws in each half-chain.
    reason : str or None
        Why ``r_hat`` is ``None``. ``None`` exactly when ``r_hat`` is set.
    """

    r_hat: float | None
    n_split_chains: int
    draws_per_split_chain: int
    reason: str | None

    def __post_init__(self) -> None:
        if (self.r_hat is None) != (self.reason is not None):
            raise ValueError("exactly one of r_hat and reason must be set")


def split_r_hat(values: torch.Tensor) -> MixingDiagnostics:
    """Return the split-Rhat mixing diagnostic for ``[draw, walker]`` samples.

    Each walker chain is split into two contiguous halves, giving
    ``2 * n_walkers`` sequences. Rhat compares the variance of the half-chain
    means against the mean of the half-chain variances; it approaches ``1`` as
    the halves become indistinguishable.

    Parameters
    ----------
    values : torch.Tensor
        Observable samples, shape ``[n_draws, n_walkers]``.

    Returns
    -------
    MixingDiagnostics
        The diagnostic, or an unresolved outcome with a reason. Odd draw counts
        drop the oldest draw so both halves are the same length.

    Raises
    ------
    ValueError
        If `values` is not a non-empty two-dimensional tensor.
    """

    if values.ndim != 2:
        raise ValueError(f"values must be [draw, walker]; got shape {tuple(values.shape)}")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"values must be non-empty; got shape {tuple(values.shape)}")

    checked = values.detach().to(torch.float64)
    n_draws, n_walkers = int(checked.shape[0]), int(checked.shape[1])
    half = n_draws // 2
    n_split_chains = 2 * n_walkers

    def unresolved(reason: str) -> MixingDiagnostics:
        return MixingDiagnostics(
            r_hat=None,
            n_split_chains=n_split_chains,
            draws_per_split_chain=half,
            reason=reason,
        )

    if half < 2:
        return unresolved(f"insufficient draws for split-Rhat: {n_draws} per chain, need at least 4")
    if not bool(torch.isfinite(checked).all()):
        return unresolved("trajectory contains non-finite values")

    # Drop the oldest draw on an odd count so both halves have `half` draws.
    trimmed = checked[n_draws - 2 * half :]
    # [2, half, n_walkers] -> [half, 2 * n_walkers]: the two halves of each
    # walker become independent columns.
    halves = trimmed.reshape(2, half, n_walkers).permute(1, 0, 2).reshape(half, n_split_chains)

    chain_means = halves.mean(dim=0)
    chain_variances = halves.var(dim=0, unbiased=True)
    within = float(chain_variances.mean().item())
    between = float((half * chain_means.var(unbiased=True)).item())

    if not (within > 0.0):
        return unresolved("all split chains are constant; split-Rhat is undefined")

    # Marginal posterior variance estimate: the within-chain variance corrected
    # upward by whatever the chain means disagree about.
    var_plus = ((half - 1) / half) * within + between / half
    r_hat = float((var_plus / within) ** 0.5)
    if not math.isfinite(r_hat):
        return unresolved("split-Rhat is not finite")

    return MixingDiagnostics(
        r_hat=r_hat,
        n_split_chains=n_split_chains,
        draws_per_split_chain=half,
        reason=None,
    )
