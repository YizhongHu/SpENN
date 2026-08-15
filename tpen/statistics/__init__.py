"""Correlation-aware statistics for fixed-model Markov-chain trajectories.

This package owns the *draw axis*. Everything here operates on trajectories
laid out as ``[draw, walker, ...]``, where walker columns are independent
chains that are never concatenated along the draw axis. Concatenating chains
manufactures a step discontinuity at every chain boundary and inflates the
apparent autocorrelation without bound, so the estimators below pool
per-chain autocovariances instead.

Three quantities are produced, and they answer different questions:

``tau_int``
    Integrated autocorrelation time of one observable, published in the
    unambiguous convention ``tau_int = 1 + 2 * sum_{k>=1} rho_k``. A bare
    ``tau`` is never published, because the half-IAT convention differs by a
    factor of two.
``ess``
    Effective sample size, ``total_draws / tau_int``. ESS answers "how many
    independent draws is this worth"; it is *not* evidence of stationarity.
``mcse``
    Monte-Carlo standard error of the mean, ``sqrt(variance / ess)``. This is
    the correlation-aware companion to the IID ``sigma / sqrt(N)`` that the
    evaluation summaries continue to publish for like-for-like comparison.

Mixing is deliberately a *separate* diagnostic (:mod:`tpen.statistics.mixing`).
A short, well-mixed chain and a long, badly-mixed chain fail in different ways,
and folding one into the other hides both. A trajectory that cannot resolve a
plateau, or whose chains disagree, yields status ``unresolved`` with a reason --
never a zero, a bound presented as a point estimate, or an imputed value.
"""

from __future__ import annotations

from tpen.statistics.autocorrelation import (
    IntegratedAutocorrelation,
    TAU_CONVENTION,
    integrated_autocorrelation_time,
    per_chain_integrated_autocorrelation,
    pooled_autocorrelation,
)
from tpen.statistics.mixing import MixingDiagnostics, split_r_hat
from tpen.statistics.producer import (
    DEFAULT_MIN_DRAWS_PER_CHAIN,
    DEFAULT_R_HAT_THRESHOLD,
    ESTIMATOR_ID,
    ESTIMATOR_VERSION,
    produce_trajectory_statistics,
)
from tpen.statistics.receipt import (
    ChainStatistics,
    PlateauDiagnostics,
    TrajectoryShape,
    TrajectoryStatisticsIdentity,
    TrajectoryStatisticsPayload,
    TrajectoryStatisticsReceipt,
    TrajectoryStatisticsStatus,
)
from tpen.statistics.sidecar import (
    DuplicateReceiptError,
    TrajectoryStatisticsSidecar,
)
from tpen.statistics.trajectory import ObservableTrajectory

__all__ = [
    "DEFAULT_MIN_DRAWS_PER_CHAIN",
    "DEFAULT_R_HAT_THRESHOLD",
    "ESTIMATOR_ID",
    "ESTIMATOR_VERSION",
    "ChainStatistics",
    "DuplicateReceiptError",
    "IntegratedAutocorrelation",
    "MixingDiagnostics",
    "ObservableTrajectory",
    "PlateauDiagnostics",
    "TAU_CONVENTION",
    "TrajectoryShape",
    "TrajectoryStatisticsIdentity",
    "TrajectoryStatisticsPayload",
    "TrajectoryStatisticsReceipt",
    "TrajectoryStatisticsSidecar",
    "TrajectoryStatisticsStatus",
    "integrated_autocorrelation_time",
    "per_chain_integrated_autocorrelation",
    "pooled_autocorrelation",
    "produce_trajectory_statistics",
    "split_r_hat",
]
