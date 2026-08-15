"""Chain-pooled autocorrelation and integrated autocorrelation time.

The estimator here follows Geyer's initial positive sequence, applied to an
autocorrelation function pooled across independent walker chains. Two design
choices are load-bearing and are not incidental:

**Per-chain means.** Each chain is centred on its *own* mean before the
autocovariance is accumulated. Centring on a global mean would leak between-chain
mean differences into every lag and report a badly-mixed ensemble as a
slowly-decaying one. Mixing is a separate question, answered separately by
:func:`tpen.statistics.mixing.split_r_hat`.

**No concatenation.** Chains are pooled by averaging their autocovariances at
matched lags, never by joining them end to end. Concatenation manufactures a
step at each boundary; for two chains whose means differ by more than their
within-chain spread, the joined series looks like a single slow chain and the
apparent ``tau_int`` grows with the chain length rather than converging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "TAU_CONVENTION",
    "IntegratedAutocorrelation",
    "integrated_autocorrelation_time",
    "pooled_autocorrelation",
]

TAU_CONVENTION = "tau_int = 1 + 2 * sum_{k>=1} rho_k"
"""The published convention, spelled out so no consumer has to guess.

The half-IAT convention differs by a factor of two. Emitting a bare ``tau``
without this string is how two correct implementations disagree by 2x.
"""

_MIN_DRAWS_PER_CHAIN = 8
"""Below this, the lag-1 estimate alone is noise and no plateau is meaningful."""


@dataclass(frozen=True)
class IntegratedAutocorrelation:
    """Outcome of an integrated-autocorrelation-time estimate.

    Attributes
    ----------
    tau_int : float or None
        Integrated autocorrelation time in the :data:`TAU_CONVENTION` sense, or
        ``None`` when the estimate did not resolve. ``None`` is never replaced
        by a zero, a one, or a bound presented as a point estimate.
    variance : float or None
        Pooled within-chain sample variance of the observable, used as the
        numerator of the Monte-Carlo standard error. ``None`` when unresolved.
    plateau_reached : bool
        Whether Geyer's initial positive sequence terminated inside the
        available lags. ``False`` means the chain is too short to see the
        autocorrelation function turn over, not that there is no correlation.
    truncation_lag : int or None
        Largest lag ``k`` included in the sum. Pairs ``m = 0 .. pair_count - 1``
        cover lags ``0 .. 2 * pair_count - 1``, so this is
        ``2 * pair_count - 1``.
    pair_count : int or None
        Number of Geyer pairs summed.
    max_lag : int
        Largest lag that could have been examined given the draw count.
    reason : str or None
        Why the estimate did not resolve. ``None`` exactly when ``tau_int`` is
        not ``None``.
    """

    tau_int: float | None
    variance: float | None
    plateau_reached: bool
    truncation_lag: int | None
    pair_count: int | None
    max_lag: int
    reason: str | None

    def __post_init__(self) -> None:
        resolved = self.tau_int is not None
        if resolved == (self.reason is not None):
            raise ValueError("exactly one of tau_int and reason must be set")


def _validated_values(values: torch.Tensor) -> torch.Tensor:
    """Return a detached ``float64`` ``[draw, walker]`` view of `values`."""

    if values.ndim != 2:
        raise ValueError(f"values must be [draw, walker]; got shape {tuple(values.shape)}")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"values must be non-empty; got shape {tuple(values.shape)}")
    return values.detach().to(torch.float64)


def _autocovariance_reference(centered: torch.Tensor) -> torch.Tensor:
    """Return per-chain autocovariances by direct summation.

    This is the readable definition, kept as the correctness reference for the
    FFT path. It is ``O(n_draws^2)`` and is not used on production-sized
    trajectories.

    Parameters
    ----------
    centered : torch.Tensor
        Per-chain mean-centred values, shape ``[n_draws, n_walkers]``.

    Returns
    -------
    torch.Tensor
        Biased autocovariance estimates, shape ``[n_draws, n_walkers]``, where
        row ``k`` is ``(1 / n_draws) * sum_t y[t] * y[t + k]``.
    """

    n_draws = centered.shape[0]
    lags = []
    for lag in range(n_draws):
        # Biased normalisation (divide by n_draws, not n_draws - lag): the
        # biased estimator is what keeps the summed sequence positive-definite,
        # which is exactly what Geyer's truncation rule relies on.
        lags.append((centered[: n_draws - lag] * centered[lag:]).sum(dim=0) / n_draws)
    return torch.stack(lags, dim=0)


def _autocovariance_fft(centered: torch.Tensor) -> torch.Tensor:
    """Return per-chain autocovariances via the Wiener-Khinchin theorem.

    Mathematically identical to :func:`_autocovariance_reference` but
    ``O(n log n)``. The transform length is padded past ``2 * n_draws`` so the
    circular correlation does not wrap onto itself.

    Parameters
    ----------
    centered : torch.Tensor
        Per-chain mean-centred values, shape ``[n_draws, n_walkers]``.

    Returns
    -------
    torch.Tensor
        Biased autocovariance estimates, shape ``[n_draws, n_walkers]``.
    """

    n_draws = centered.shape[0]
    # Next power of two at or above 2 * n_draws: enough zero padding to make the
    # circular correlation linear, and a length the FFT handles efficiently.
    n_fft = 1 << max(1, 2 * n_draws - 1).bit_length()
    spectrum = torch.fft.rfft(centered, n=n_fft, dim=0)
    power = spectrum.real.square() + spectrum.imag.square()
    return torch.fft.irfft(power, n=n_fft, dim=0)[:n_draws] / n_draws


def pooled_autocorrelation(values: torch.Tensor, *, method: str = "fft") -> torch.Tensor:
    """Return the chain-pooled autocorrelation function.

    Each chain is centred on its own mean, its autocovariance is accumulated
    independently, and the per-chain autocovariances are averaged at matched
    lags. Chains are never concatenated.

    Parameters
    ----------
    values : torch.Tensor
        Observable samples, shape ``[n_draws, n_walkers]``.
    method : {'fft', 'reference'}, optional
        ``'fft'`` uses the Wiener-Khinchin transform; ``'reference'`` uses the
        direct ``O(n^2)`` definition. They agree to floating-point tolerance;
        ``'reference'`` exists so the fast path has something to be tested
        against.

    Returns
    -------
    torch.Tensor
        Autocorrelation ``rho`` of shape ``[n_draws]`` with ``rho[0] == 1``.
        Returns a length-``n_draws`` vector of zeros with ``rho[0] == 0`` when
        the pooled variance vanishes, which the caller must treat as
        unresolved rather than as perfect independence.

    Raises
    ------
    ValueError
        If `values` is not a non-empty two-dimensional tensor, or `method` is
        not recognised.
    """

    checked = _validated_values(values)
    centered = checked - checked.mean(dim=0, keepdim=True)
    if method == "fft":
        autocovariance = _autocovariance_fft(centered)
    elif method == "reference":
        autocovariance = _autocovariance_reference(centered)
    else:
        raise ValueError(f"method must be 'fft' or 'reference', got {method!r}")

    # Pool across chains at matched lags. Averaging autocovariances -- rather
    # than averaging per-chain autocorrelations -- weights each chain by its own
    # variance, which is the behaviour wanted when one walker is stuck.
    pooled = autocovariance.mean(dim=1)
    variance = pooled[0]
    if not bool(torch.isfinite(variance)) or float(variance) <= 0.0:
        return torch.zeros_like(pooled)
    return pooled / variance


def integrated_autocorrelation_time(
    values: torch.Tensor,
    *,
    method: str = "fft",
    min_draws_per_chain: int = _MIN_DRAWS_PER_CHAIN,
) -> IntegratedAutocorrelation:
    """Estimate ``tau_int`` with Geyer's initial positive sequence.

    Consecutive autocorrelation lags are summed in pairs,
    ``Gamma_m = rho_{2m} + rho_{2m+1}``. For a reversible Markov chain this
    paired sequence is positive and non-increasing; the estimator sums pairs
    until one turns non-positive, which is the point where the remaining signal
    is indistinguishable from sampling noise. Running the sum past that point is
    the classic way to accumulate unbounded variance from the noisy tail.

    A monotone-decreasing envelope is applied to the pairs before summation, so
    a single noisy upward excursion cannot inflate the result.

    Parameters
    ----------
    values : torch.Tensor
        Observable samples, shape ``[n_draws, n_walkers]``.
    method : {'fft', 'reference'}, optional
        Autocovariance backend, forwarded to :func:`pooled_autocorrelation`.
    min_draws_per_chain : int, optional
        Refuse to estimate below this many draws per chain.

    Returns
    -------
    IntegratedAutocorrelation
        Resolved estimate, or an unresolved outcome carrying the reason. A
        chain that is too short, constant, non-finite, or that never turns over
        yields ``tau_int=None``; it never yields a fabricated number.
    """

    checked = _validated_values(values)
    n_draws, n_walkers = int(checked.shape[0]), int(checked.shape[1])
    max_lag = max(n_draws - 1, 0)

    def unresolved(reason: str) -> IntegratedAutocorrelation:
        return IntegratedAutocorrelation(
            tau_int=None,
            variance=None,
            plateau_reached=False,
            truncation_lag=None,
            pair_count=None,
            max_lag=max_lag,
            reason=reason,
        )

    if not bool(torch.isfinite(checked).all()):
        # Dropping non-finite draws would silently re-index every later lag, so
        # a contaminated trajectory is refused rather than repaired.
        return unresolved("trajectory contains non-finite values")
    if n_draws < min_draws_per_chain:
        return unresolved(
            f"insufficient draws: {n_draws} per chain across {n_walkers} chain(s), "
            f"minimum {min_draws_per_chain}"
        )

    rho = pooled_autocorrelation(checked, method=method)
    if float(rho[0]) <= 0.0:
        return unresolved("observable has zero pooled variance across all chains")

    # Geyer pairs. Pair m spans lags 2m and 2m + 1, so the last complete pair is
    # bounded by the largest odd lag available, 2m + 1 <= n_draws - 1. That gives
    # n_draws // 2 pairs: `(n_draws - 1) // 2` silently discards the final
    # complete pair on even draw counts, which shortens the window examined and
    # can turn a resolvable plateau into a spurious "no plateau".
    n_pairs = n_draws // 2
    if n_pairs < 1:
        return unresolved(f"insufficient draws to form a Geyer pair: {n_draws} per chain")
    pairs = rho[: 2 * n_pairs].reshape(n_pairs, 2).sum(dim=1)
    # Monotone envelope: cumulative running minimum over the pair sequence.
    monotone = torch.cummin(pairs, dim=0).values

    positive = monotone > 0.0
    if not bool(positive[0]):
        # The very first pair is already non-positive: strongly anti-correlated
        # or pure noise at this length. Either way there is nothing to sum.
        return unresolved("initial Geyer pair is non-positive; autocorrelation did not resolve")
    if bool(positive.all()):
        # Never turned over inside the available lags. The true tau_int is at
        # least the partial sum, but a lower bound is not a point estimate.
        return unresolved(
            f"no plateau within {max_lag} lags; trajectory too short to bound tau_int"
        )

    # Index of the first non-positive pair. `nonzero` returns ascending indices,
    # so this is unambiguous; `argmin` would rely on tie-breaking behaviour that
    # torch does not guarantee across backends.
    pair_count = int((~positive).nonzero()[0].item())
    # tau_int = -1 + 2 * sum(Gamma_m) is algebraically 1 + 2 * sum_{k>=1} rho_k,
    # because sum(Gamma_m) telescopes to rho_0 + sum_{k>=1} rho_k and rho_0 == 1.
    tau_int = float(-1.0 + 2.0 * monotone[:pair_count].sum().item())
    # `not (tau_int > 0.0)` also rejects NaN, which no comparison would accept.
    if not math.isfinite(tau_int) or not (tau_int > 0.0):
        return unresolved(f"non-positive tau_int estimate: {tau_int}")

    # Unbiased pooled within-chain variance for the standard error. This uses
    # the (n - 1) denominator, unlike the biased autocovariance used for rho.
    variance = float(checked.var(dim=0, unbiased=True).mean().item())
    if not (variance > 0.0):
        return unresolved("observable has zero within-chain variance")

    return IntegratedAutocorrelation(
        tau_int=tau_int,
        variance=variance,
        plateau_reached=True,
        truncation_lag=2 * pair_count - 1,
        pair_count=pair_count,
        max_lag=max_lag,
        reason=None,
    )
