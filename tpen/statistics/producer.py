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
)
from tpen.statistics.mixing import split_r_hat
from tpen.statistics.receipt import (
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
    source_sha256 = trajectory.content_sha256

    autocorrelation = integrated_autocorrelation_time(
        trajectory.values,
        method=method,
        min_draws_per_chain=min_draws_per_chain,
    )
    plateau = PlateauDiagnostics(
        plateau_reached=autocorrelation.plateau_reached,
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
            source_artifact_sha256=source_sha256,
            reason=reason,
            warnings=warnings,
        )

    warnings: list[str] = []
    if trajectory.nonfinite_count:
        # Reported rather than filtered: removing entries from a time series
        # shifts every subsequent lag and corrupts the autocorrelation.
        return unresolved(
            f"trajectory contains {trajectory.nonfinite_count} non-finite value(s); "
            "non-finite draws are never dropped because removing them re-indexes every lag"
        )

    if autocorrelation.tau_int is None or autocorrelation.variance is None:
        # IntegratedAutocorrelation guarantees a reason whenever tau_int is None;
        # the fallback keeps the receipt valid even if that ever regresses.
        return unresolved(autocorrelation.reason or "autocorrelation did not resolve")

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

    total_draws = shape.total_draws
    ess = total_draws / autocorrelation.tau_int
    if not (ess > 0.0):
        return unresolved(f"non-positive effective sample size: {ess}", warnings=tuple(warnings))
    # Correlation-aware standard error: the IID sigma/sqrt(N) with N replaced by
    # the effective sample size. When tau_int == 1 the two coincide.
    mcse = (autocorrelation.variance / ess) ** 0.5

    if ess < 1.0:
        warnings.append(
            f"effective sample size {ess:.3f} is below one draw; the mean is not resolved"
        )
    # ESS is a precision statement, never a stationarity statement. Say so on
    # every receipt so no consumer can read a healthy ESS as convergence.
    warnings.append("ESS quantifies precision only and is not evidence of stationarity")

    payload = TrajectoryStatisticsPayload(
        tau_int=autocorrelation.tau_int,
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
        source_artifact_sha256=source_sha256,
        warnings=tuple(warnings),
    )
