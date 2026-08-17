"""Turn a fixed-model trajectory into an immutable statistics receipt.

This is the single sanctioned producer of ``tau_int``, ``ess`` and ``mcse`` in
TPEN. Downstream consumers -- evaluation summaries, the experiment toolkit's
cost projection -- validate and report what they find here; they never
re-estimate it, and in particular they never infer autocorrelation from wall-
clock or event durations, which measure how long a step took rather than how
correlated it was with its predecessor.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from tpen.checkpoint.hashing import file_sha256
from tpen.statistics.autocorrelation import (
    TAU_CONVENTION,
    integrated_autocorrelation_time,
    per_chain_integrated_autocorrelation,
)
from tpen.statistics.mixing import split_r_hat
from tpen.statistics.receipt import (
    ChainStatistics,
    PlateauDiagnostics,
    TrajectoryShape,
    TrajectoryStatisticsIdentity,
    TrajectoryStatisticsPayload,
    TrajectoryStatisticsReceipt,
)
from tpen.statistics.trajectory import ObservableTrajectory

__all__ = [
    "DEFAULT_MIN_DRAWS_PER_CHAIN",
    "DEFAULT_R_HAT_THRESHOLD",
    "DEFAULT_R_HAT_WARN_THRESHOLD",
    "ESTIMATOR_ID",
    "ESTIMATOR_VERSION",
    "absent_receipt",
    "produce_trajectory_statistics",
]

ESTIMATOR_ID = "pooled_geyer_ips"
"""Chain-pooled autocovariance with Geyer initial-positive-sequence truncation."""

ESTIMATOR_VERSION = "1"
"""Bumped whenever the numbers this estimator produces would change."""

DEFAULT_MIN_DRAWS_PER_CHAIN = 8
"""Fewer draws than this cannot distinguish a plateau from noise."""

DEFAULT_R_HAT_WARN_THRESHOLD = 1.01
"""Above this, chains disagree enough to be worth saying out loud."""

DEFAULT_R_HAT_THRESHOLD = 1.1
"""Above this, the mean itself is not trustworthy, so no payload is published.

ESS and Rhat are computed independently -- Rhat never enters ``tau_int`` -- but a
correlation-aware error bar around a mean that the chains do not agree on would
be precise and wrong. The receipt therefore reports ``unresolved`` with the
measured Rhat in its reason rather than publishing a confident interval.
"""


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def absent_receipt(
    identity: TrajectoryStatisticsIdentity,
    *,
    reason: str,
    recorded_at_utc: str | None = None,
) -> TrajectoryStatisticsReceipt:
    """Return an ``absent`` receipt for an identity with no trajectory.

    An explicit ``absent`` row is not the same as a missing row: it records that
    the question was asked and that nothing was collected, which is what stops a
    consumer from quietly treating the gap as a zero.

    Parameters
    ----------
    identity : TrajectoryStatisticsIdentity
        The join key the missing trajectory would have had.
    reason : str
        Why no trajectory exists.
    recorded_at_utc : str or None, optional
        Override the timestamp; defaults to now.

    Returns
    -------
    TrajectoryStatisticsReceipt
        A receipt with ``status="absent"`` and no payload.
    """

    return TrajectoryStatisticsReceipt(
        identity=identity,
        status="absent",
        recorded_at_utc=recorded_at_utc or _utc_now(),
        estimator_id=ESTIMATOR_ID,
        estimator_version=ESTIMATOR_VERSION,
        tau_convention=TAU_CONVENTION,
        reason=reason,
    )


def produce_trajectory_statistics(
    trajectory: ObservableTrajectory,
    identity: TrajectoryStatisticsIdentity,
    *,
    min_draws_per_chain: int = DEFAULT_MIN_DRAWS_PER_CHAIN,
    r_hat_threshold: float = DEFAULT_R_HAT_THRESHOLD,
    r_hat_warn_threshold: float = DEFAULT_R_HAT_WARN_THRESHOLD,
    method: str = "fft",
    checkpoint_path: Path | str | None = None,
    recorded_at_utc: str | None = None,
) -> TrajectoryStatisticsReceipt:
    """Produce the immutable statistics receipt for one observable trajectory.

    Parameters
    ----------
    trajectory : ObservableTrajectory
        Fixed-model samples with explicit ``[draw, walker]`` axes.
    identity : TrajectoryStatisticsIdentity
        The seven-part join key. Its ``observable`` must match the
        trajectory's, since a receipt filed under the wrong observable joins
        cleanly and means nothing.
    min_draws_per_chain : int, optional
        Refuse to estimate below this many draws per chain.
    r_hat_threshold : float, optional
        Split-Rhat at or above which the receipt is ``unresolved``.
    r_hat_warn_threshold : float, optional
        Split-Rhat above which a warning is attached but a payload is still
        published.
    method : {'fft', 'reference'}, optional
        Autocovariance backend.
    checkpoint_path : Path or str or None, optional
        Checkpoint file whose contents must hash to
        ``identity.checkpoint_sha256``. When given, the claim is verified rather
        than trusted, so samples from one model cannot be filed under another
        model's key. Omit only when the checkpoint file is genuinely
        unavailable to the caller.
    recorded_at_utc : str or None, optional
        Override the timestamp; defaults to now.

    Returns
    -------
    TrajectoryStatisticsReceipt
        ``available`` with a full payload, or ``unresolved`` with a reason.
        Never a partially-filled row, and never an imputed zero.

    Raises
    ------
    ValueError
        If ``identity.observable`` does not match ``trajectory.observable``, or
        ``checkpoint_path`` is given and its contents do not hash to
        ``identity.checkpoint_sha256``.
    """

    if identity.observable != trajectory.observable:
        raise ValueError(
            "identity.observable does not match the trajectory: "
            f"{identity.observable!r} != {trajectory.observable!r}. "
            "Autocorrelation is observable-specific; a mismatched receipt would "
            "join onto the wrong measurement."
        )

    # The identity's checkpoint_sha256 is caller-supplied, so on its own it
    # asserts provenance rather than establishing it: nothing stops model-B
    # samples being filed under checkpoint-A's key, and the sidecar would then
    # hold a well-formed receipt joining onto the wrong model. When the caller
    # can name the checkpoint file, bind the claim to its actual contents.
    if checkpoint_path is not None:
        actual = file_sha256(checkpoint_path)
        if actual != identity.checkpoint_sha256:
            raise ValueError(
                "identity.checkpoint_sha256 does not match the checkpoint contents: "
                f"claimed {identity.checkpoint_sha256}, found {actual} at "
                f"{checkpoint_path}. The join key is content-addressed precisely so "
                "that this cannot pass silently."
            )

    timestamp = recorded_at_utc or _utc_now()
    shape = TrajectoryShape(
        walker_count=trajectory.n_walkers,
        draw_count=trajectory.n_draws,
        draw_stride=trajectory.draw_stride,
        burn_in_draws=trajectory.burn_in_draws,
    )

    # Per-walker first, pooled second. The IPS truncation is nonlinear, so
    # pooling autocovariances before the decision lets one chain that never
    # plateaus be absorbed into a well-behaved average and still yield a
    # confident number. Deciding per chain keeps `unresolved` reachable.
    per_chain = per_chain_integrated_autocorrelation(
        trajectory.values,
        method=method,
        min_draws_per_chain=min_draws_per_chain,
    )
    chain_draws = trajectory.n_draws
    chain_means = trajectory.values.mean(dim=0)
    chains = tuple(
        ChainStatistics(
            index=index,
            n_draws=chain_draws,
            status="available" if result.tau_int is not None else "unresolved",
            tau_int=result.tau_int,
            plateau_reached=result.plateau_reached,
            mean=float(chain_means[index].item()) if result.tau_int is not None else None,
            variance=result.variance if result.tau_int is not None else None,
            reason=result.reason,
        )
        for index, result in enumerate(per_chain)
    )
    # The auxiliary matched-lag pooled estimator is retained for the plateau
    # block only; it never supplies tau, ESS, or MCSE.
    autocorrelation = integrated_autocorrelation_time(
        trajectory.values,
        method=method,
        min_draws_per_chain=min_draws_per_chain,
    )
    plateau = PlateauDiagnostics(
        plateau_reached=all(chain.plateau_reached for chain in chains),
        truncation_lag=autocorrelation.truncation_lag,
        pair_count=autocorrelation.pair_count,
        max_lag=autocorrelation.max_lag,
    )
    mixing = split_r_hat(trajectory.values)

    def unresolved(reason: str, warnings: tuple[str, ...] = ()) -> TrajectoryStatisticsReceipt:
        return TrajectoryStatisticsReceipt(
            identity=identity,
            status="unresolved",
            recorded_at_utc=timestamp,
            estimator_id=ESTIMATOR_ID,
            estimator_version=ESTIMATOR_VERSION,
            tau_convention=TAU_CONVENTION,
            shape=shape,
            plateau=plateau,
            mixing=mixing,
            reason=reason,
            warnings=warnings,
            chains=chains,
        )

    warnings: list[str] = []
    if trajectory.nonfinite_count:
        # Reported rather than filtered: removing entries from a time series
        # shifts every subsequent lag and corrupts the autocorrelation.
        return unresolved(
            f"trajectory contains {trajectory.nonfinite_count} non-finite value(s); "
            "non-finite draws are never dropped because removing them re-indexes every lag"
        )

    unresolved_chains = tuple(chain for chain in chains if chain.status != "available")
    if unresolved_chains:
        # Every failing chain is named. Dropping them and pooling the survivors
        # would silently redefine the estimand as "the walkers that behaved",
        # which is a different and flattering quantity.
        detail = "; ".join(
            f"chain {chain.index}: {chain.reason}" for chain in unresolved_chains
        )
        return unresolved(
            f"{len(unresolved_chains)} of {len(chains)} chain(s) did not resolve "
            f"their own integrated autocorrelation time -- {detail}"
        )

    if mixing.r_hat is None:
        warnings.append(f"split-Rhat unavailable: {mixing.reason}")
    elif mixing.r_hat >= r_hat_threshold:
        return unresolved(
            f"chains not mixed: split-Rhat {mixing.r_hat:.4f} >= {r_hat_threshold}. "
            "tau_int and ESS were computed but a correlation-aware error bar around a "
            "disputed mean would be precise and wrong",
            warnings=tuple(warnings),
        )
    elif mixing.r_hat > r_hat_warn_threshold:
        warnings.append(
            f"split-Rhat {mixing.r_hat:.4f} exceeds {r_hat_warn_threshold}; "
            "chains agree only approximately"
        )

    # Pool only resolved per-chain estimates, weighting each chain by the draws
    # it contributed. Writing the weights explicitly rather than assuming equal
    # chains keeps the algebra honest if ragged trajectories ever arrive; today
    # every column has the same length, so w_i == 1 / n_chains.
    total_draws = shape.total_draws
    ess = sum(chain.n_draws / chain.tau_int for chain in chains)
    if not (ess > 0.0):
        return unresolved(f"non-positive effective sample size: {ess}", warnings=tuple(warnings))

    # Var(sum_i w_i * mean_i) = sum_i w_i^2 * Var(mean_i), and for a correlated
    # chain Var(mean_i) = s_i^2 * tau_i / N_i. The tempting shortcut
    # sqrt(mean(s_i^2) / ess) is a different quantity: it assumes every chain
    # shares one variance and one tau, so it is wrong exactly when the walkers
    # disagree, which is when an honest error bar matters most.
    mcse_squared = sum(
        (chain.n_draws / total_draws) ** 2 * chain.variance * chain.tau_int / chain.n_draws
        for chain in chains
    )
    mcse = mcse_squared**0.5

    # A scalar tau is reported only as the value consistent with the pooled ESS,
    # never as an independently estimated quantity. Per-chain tau_i are retained
    # on the receipt so this reduction is auditable rather than authoritative.
    tau_pooled = total_draws / ess

    if ess < 1.0:
        warnings.append(
            f"effective sample size {ess:.3f} is below one draw; the mean is not resolved"
        )
    tau_values = [chain.tau_int for chain in chains]
    if len(tau_values) > 1 and min(tau_values) > 0.0:
        spread = max(tau_values) / min(tau_values)
        if spread >= 2.0:
            warnings.append(
                f"per-chain tau_int spans a factor of {spread:.2f} "
                f"(min {min(tau_values):.3f}, max {max(tau_values):.3f}); "
                "walkers are not exploring at the same rate"
            )
    # ESS is a precision statement, never a stationarity statement. Say so on
    # every receipt so no consumer can read a healthy ESS as convergence.
    warnings.append("ESS quantifies precision only and is not evidence of stationarity")

    payload = TrajectoryStatisticsPayload(
        tau_int=tau_pooled,
        ess=ess,
        mcse=mcse,
        mean=float(trajectory.values.mean().item()),
        variance=autocorrelation.variance,
    )
    return TrajectoryStatisticsReceipt(
        identity=identity,
        status="available",
        recorded_at_utc=timestamp,
        estimator_id=ESTIMATOR_ID,
        estimator_version=ESTIMATOR_VERSION,
        tau_convention=TAU_CONVENTION,
        shape=shape,
        plateau=plateau,
        mixing=mixing,
        payload=payload,
        warnings=tuple(warnings),
        chains=chains,
    )
