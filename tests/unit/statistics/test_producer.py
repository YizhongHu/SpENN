"""End-to-end contract tests for trajectory-statistics production."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest
import torch

from tpen.checkpoint.hashing import file_sha256
from tpen.statistics.autocorrelation import TAU_CONVENTION
from tpen.statistics.producer import (
    DEFAULT_R_HAT_THRESHOLD,
    DEFAULT_R_HAT_WARN_THRESHOLD,
    ESTIMATOR_ID,
    ESTIMATOR_VERSION,
    absent_receipt,
    produce_trajectory_statistics,
)
from tpen.statistics.receipt import TrajectoryStatisticsIdentity
from tpen.statistics.trajectory import ObservableTrajectory


def _identity(*, observable: str = "local_energy") -> TrajectoryStatisticsIdentity:
    """Build a complete trajectory-statistics join identity.

    Parameters
    ----------
    observable : str, optional
        Observable named by the receipt.

    Returns
    -------
    TrajectoryStatisticsIdentity
        Valid identity with deterministic content hashes.
    """

    return TrajectoryStatisticsIdentity(
        stage="evaluation",
        run_id="he-seeded-run",
        attempt_id="attempt-1",
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        observable=observable,
        evaluator_id="local_energy/v1",
    )


def _ar1_trajectory(
    *,
    seed: int,
    n_draws: int = 512,
    n_walkers: int = 6,
    phi: float = 0.7,
    observable: str = "local_energy",
    draw_stride: int = 3,
    burn_in_draws: int = 11,
) -> ObservableTrajectory:
    """Build a seeded stationary AR(1) trajectory with explicit chain columns.

    Parameters
    ----------
    seed : int
        Seed for the local random generator.
    n_draws : int, optional
        Retained draws per walker.
    n_walkers : int, optional
        Number of independent walker chains.
    phi : float, optional
        Lag-one autoregressive coefficient.
    observable : str, optional
        Observable recorded on the trajectory.
    draw_stride : int, optional
        Sampler steps between retained draws.
    burn_in_draws : int, optional
        Draws discarded before the retained trajectory.

    Returns
    -------
    ObservableTrajectory
        Float64 samples laid out as ``[draw, walker]``.
    """

    generator = torch.Generator().manual_seed(seed)
    innovations = torch.randn(
        (n_draws, n_walkers),
        generator=generator,
        dtype=torch.float64,
    )
    values = torch.empty_like(innovations)
    values[0] = innovations[0]
    innovation_scale = math.sqrt(1.0 - phi**2)
    for draw in range(1, n_draws):
        values[draw] = phi * values[draw - 1] + innovation_scale * innovations[draw]

    return ObservableTrajectory(
        observable=observable,
        values=values,
        draw_stride=draw_stride,
        burn_in_draws=burn_in_draws,
    )


def test_available_receipt_preserves_end_to_end_producer_contract() -> None:
    """Publish all fields a downstream statistics consumer joins and derives."""

    trajectory = _ar1_trajectory(
        seed=101,
        n_draws=512,
        n_walkers=6,
        phi=0.75,
        draw_stride=4,
        burn_in_draws=17,
    )
    recorded_at_utc = "2026-08-15T16:30:00Z"
    receipt = produce_trajectory_statistics(
        trajectory,
        _identity(),
        recorded_at_utc=recorded_at_utc,
    )

    # Availability must be represented by a complete payload, never a reason.
    assert receipt.status == "available"
    assert receipt.payload is not None
    assert receipt.reason is None
    assert receipt.shape is not None

    payload = receipt.payload
    shape = receipt.shape
    # Assert definitions as relations between emitted fields so estimator
    # changes cannot leave internally inconsistent wire data.
    assert payload.ess == pytest.approx(shape.total_draws / payload.tau_int, rel=1e-12)
    assert payload.mcse == pytest.approx(
        math.sqrt(payload.variance / payload.ess),
        rel=1e-12,
    )
    assert payload.mean == pytest.approx(float(trajectory.values.mean().item()), rel=1e-12)

    # The spelling prevents a consumer from silently choosing the half-IAT convention.
    assert receipt.tau_convention == TAU_CONVENTION
    assert "1 + 2 * sum" in receipt.tau_convention
    assert receipt.tau_convention != "tau"
    assert receipt.estimator_id == ESTIMATOR_ID
    assert receipt.estimator_version == ESTIMATOR_VERSION
    assert receipt.source_artifact_sha256 == trajectory.content_sha256

    # Shape metadata preserves chain boundaries and sampling cadence at the join.
    assert shape.to_dict() == {
        "walker_count": trajectory.n_walkers,
        "draw_count": trajectory.n_draws,
        "total_draws": trajectory.total_draws,
        "draw_stride": trajectory.draw_stride,
        "burn_in_draws": trajectory.burn_in_draws,
    }
    assert receipt.recorded_at_utc == recorded_at_utc

    generated_timestamp = produce_trajectory_statistics(
        trajectory,
        _identity(),
    ).recorded_at_utc
    # Default timestamps must be portable UTC ISO-8601 values, not local time.
    assert generated_timestamp.endswith("Z")
    parsed_timestamp = datetime.fromisoformat(generated_timestamp.replace("Z", "+00:00"))
    assert parsed_timestamp.tzinfo == UTC


def test_mcse_exceeds_iid_standard_error_for_autocorrelated_trajectory() -> None:
    """Inflate uncertainty by the measured IAT for a correlated chain."""

    trajectory = _ar1_trajectory(
        seed=202,
        n_draws=2048,
        n_walkers=8,
        phi=0.85,
    )
    receipt = produce_trajectory_statistics(trajectory, _identity())

    assert receipt.status == "available"
    assert receipt.payload is not None
    assert receipt.shape is not None
    payload = receipt.payload
    naive_iid_stderr = math.sqrt(payload.variance / receipt.shape.total_draws)

    # Positive AR(1) persistence is exactly the case where IID error understates risk.
    assert payload.tau_int > 1.0
    assert payload.mcse > naive_iid_stderr
    assert payload.mcse / naive_iid_stderr == pytest.approx(
        math.sqrt(payload.tau_int),
        rel=1e-12,
    )


def test_available_receipt_warns_that_ess_does_not_prove_stationarity() -> None:
    """Keep the stationarity caveat on every healthy payload."""

    receipt = produce_trajectory_statistics(
        _ar1_trajectory(seed=303, phi=0.5),
        _identity(),
    )

    assert receipt.status == "available"
    assert receipt.payload is not None
    # A downstream report must not render a healthy ESS as a convergence claim.
    assert (
        "ESS quantifies precision only and is not evidence of stationarity"
        in receipt.warnings
    )


def test_insufficient_draws_is_unresolved_without_imputed_statistics() -> None:
    """Preserve explanatory metadata while withholding unresolvable numbers."""

    trajectory = _ar1_trajectory(seed=404, n_draws=7, n_walkers=4)
    receipt = produce_trajectory_statistics(trajectory, _identity())
    serialized = receipt.to_dict()

    assert receipt.status == "unresolved"
    assert receipt.payload is None
    assert receipt.reason
    # None prevents missing estimates from being consumed as measured zeros.
    assert serialized["statistics"] is None
    # Shape and plateau metadata explain why the estimator refused the trajectory.
    assert receipt.shape is not None
    assert receipt.plateau is not None
    assert serialized["shape"] is not None
    assert serialized["plateau"] is not None
    assert receipt.shape.draw_count == trajectory.n_draws


@pytest.mark.parametrize(
    "nonfinite",
    [float("nan"), float("inf")],
    ids=["nan", "inf"],
)
def test_nonfinite_trajectory_is_unresolved_without_dropping_draws(nonfinite: float) -> None:
    """Reject non-finite samples without changing the trajectory lag index."""

    base = _ar1_trajectory(seed=505, n_draws=128, n_walkers=4)
    values = base.values.clone()
    values[13, 2] = nonfinite
    trajectory = ObservableTrajectory(
        observable=base.observable,
        values=values,
        draw_stride=base.draw_stride,
        burn_in_draws=base.burn_in_draws,
    )
    receipt = produce_trajectory_statistics(trajectory, _identity())

    assert receipt.status == "unresolved"
    assert receipt.payload is None
    assert receipt.reason is not None
    assert "non-finite" in receipt.reason.lower()
    assert receipt.shape is not None
    # Keeping the full count proves contamination was reported, not filtered and re-indexed.
    assert receipt.shape.draw_count == trajectory.n_draws == values.shape[0]
    assert receipt.shape.total_draws == trajectory.total_draws


def test_split_rhat_gate_rejects_separated_chains_and_threshold_admits_them() -> None:
    """Make payload availability depend specifically on the split-Rhat gate."""

    base = _ar1_trajectory(seed=606, n_draws=512, n_walkers=2, phi=0.6)
    offsets = torch.tensor([-6.0, 6.0], dtype=torch.float64)
    trajectory = ObservableTrajectory(
        observable=base.observable,
        values=base.values + offsets,
        draw_stride=base.draw_stride,
        burn_in_draws=base.burn_in_draws,
    )

    rejected = produce_trajectory_statistics(trajectory, _identity())

    assert rejected.status == "unresolved"
    assert rejected.payload is None
    assert rejected.reason is not None
    assert "split-Rhat" in rejected.reason
    assert rejected.mixing is not None
    assert rejected.mixing.r_hat is not None

    admitted = produce_trajectory_statistics(
        trajectory,
        _identity(),
        r_hat_threshold=rejected.mixing.r_hat + 1.0,
    )
    # Admitting identical data by changing only the threshold isolates the failing gate.
    assert admitted.status == "available"
    assert admitted.payload is not None
    assert admitted.reason is None


def test_split_rhat_warning_band_preserves_available_payload() -> None:
    """Warn between the configured split-Rhat bounds without suppressing data."""

    half_draws = 128
    n_walkers = 2
    generator = torch.Generator().manual_seed(707)
    halves = torch.randn(
        (2, half_draws, n_walkers),
        generator=generator,
        dtype=torch.float64,
    )
    # Exact zero means and unit sample variances make the target Rhat algebraic.
    halves = halves - halves.mean(dim=1, keepdim=True)
    halves = halves / halves.std(dim=1, unbiased=True, keepdim=True)
    target_r_hat = (DEFAULT_R_HAT_WARN_THRESHOLD + DEFAULT_R_HAT_THRESHOLD) / 2.0
    finite_draw_correction = (half_draws - 1) / half_draws
    offset = math.sqrt(0.75 * (target_r_hat**2 - finite_draw_correction))
    values = torch.cat((halves[0], halves[1]), dim=0)
    values = values + torch.tensor([-offset, offset], dtype=torch.float64)
    trajectory = ObservableTrajectory(
        observable="local_energy",
        values=values,
        draw_stride=2,
        burn_in_draws=9,
    )

    receipt = produce_trajectory_statistics(
        trajectory,
        _identity(),
        r_hat_warn_threshold=DEFAULT_R_HAT_WARN_THRESHOLD,
        r_hat_threshold=DEFAULT_R_HAT_THRESHOLD,
    )

    assert receipt.status == "available"
    assert receipt.payload is not None
    assert receipt.mixing is not None
    assert receipt.mixing.r_hat is not None
    # The constructed split-chain means place Rhat strictly inside the warning band.
    assert receipt.mixing.r_hat == pytest.approx(target_r_hat, rel=1e-12, abs=1e-12)
    assert DEFAULT_R_HAT_WARN_THRESHOLD < receipt.mixing.r_hat
    assert receipt.mixing.r_hat < DEFAULT_R_HAT_THRESHOLD
    assert any("split-Rhat" in warning for warning in receipt.warnings)


def test_observable_mismatch_is_rejected() -> None:
    """Refuse a receipt identity that would join onto another observable."""

    trajectory = _ar1_trajectory(seed=808, n_draws=64)

    # IAT is observable-specific, so a mismatched key is a category error.
    with pytest.raises(ValueError, match="identity.observable does not match"):
        produce_trajectory_statistics(
            trajectory,
            _identity(observable="local_energy_gradient"),
        )


def test_absent_receipt_preserves_identity_reason_and_producer_metadata() -> None:
    """Distinguish a declined production attempt from an imputed measurement."""

    identity = _identity()
    reason = "trajectory collection was disabled"
    receipt = absent_receipt(
        identity,
        reason=reason,
        recorded_at_utc="2026-08-15T18:00:00Z",
    )

    assert receipt.status == "absent"
    assert receipt.payload is None
    assert receipt.shape is None
    assert receipt.reason == reason
    assert receipt.identity == identity
    # Producer metadata lets consumers distinguish which estimator declined.
    assert receipt.estimator_id == ESTIMATOR_ID
    assert receipt.estimator_version == ESTIMATOR_VERSION
    assert receipt.tau_convention == TAU_CONVENTION
    assert receipt.to_dict()["statistics"] is None


def test_producer_is_deterministic_with_an_explicit_timestamp() -> None:
    """Emit byte-equivalent mappings for identical data and provenance."""

    trajectory = _ar1_trajectory(seed=909, n_draws=384, n_walkers=5, phi=0.65)
    identity = _identity()
    recorded_at_utc = "2026-08-15T19:00:00Z"

    first = produce_trajectory_statistics(
        trajectory,
        identity,
        recorded_at_utc=recorded_at_utc,
    )
    second = produce_trajectory_statistics(
        trajectory,
        identity,
        recorded_at_utc=recorded_at_utc,
    )

    # Explicit time removes the sole intended source of receipt nondeterminism.
    assert first.to_dict() == second.to_dict()


def test_reference_and_fft_producers_agree_on_tau_int() -> None:
    """Keep the production FFT estimator tied to the readable reference path."""

    trajectory = _ar1_trajectory(seed=1010, n_draws=256, n_walkers=4, phi=0.55)
    identity = _identity()
    reference = produce_trajectory_statistics(
        trajectory,
        identity,
        method="reference",
        recorded_at_utc="2026-08-15T20:00:00Z",
    )
    fft = produce_trajectory_statistics(
        trajectory,
        identity,
        method="fft",
        recorded_at_utc="2026-08-15T20:00:00Z",
    )

    assert reference.status == "available"
    assert fft.status == "available"
    assert reference.payload is not None
    assert fft.payload is not None
    # Float64 direct sums and FFT reductions differ only by rounding order.
    assert fft.payload.tau_int == pytest.approx(
        reference.payload.tau_int,
        rel=1e-10,
        abs=1e-12,
    )


def test_checkpoint_path_mismatch_is_rejected_before_any_statistics(tmp_path) -> None:
    """Refuse to file one model's samples under another model's identity.

    ``identity.checkpoint_sha256`` is caller-supplied, so unverified it asserts
    provenance rather than establishing it. Without this binding a receipt for
    model B can be written under model A's key: well-formed, joinable, and
    wrong. The whole join key is content-addressed to make that impossible, so
    the check must run before any statistic is computed.
    """
    checkpoint = tmp_path / "model-b.pt"
    checkpoint.write_bytes(b"model-b weights")

    with pytest.raises(ValueError, match="checkpoint_sha256 does not match"):
        produce_trajectory_statistics(
            _ar1_trajectory(seed=1301),
            _identity(),  # claims "a" * 64
            checkpoint_path=checkpoint,
        )


def test_matching_checkpoint_contents_are_accepted(tmp_path) -> None:
    """Admit the receipt when the claimed digest is the file's real digest."""
    checkpoint = tmp_path / "model-a.pt"
    checkpoint.write_bytes(b"model-a weights")
    digest = file_sha256(checkpoint)
    identity = TrajectoryStatisticsIdentity(
        stage="evaluation",
        run_id="he-seeded-run",
        attempt_id="attempt-1",
        checkpoint_sha256=digest,
        config_sha256="b" * 64,
        observable="local_energy",
        evaluator_id="local_energy/v1",
    )

    receipt = produce_trajectory_statistics(
        _ar1_trajectory(seed=1302), identity, checkpoint_path=checkpoint
    )

    assert receipt.identity.checkpoint_sha256 == digest
    assert receipt.status in ("available", "unresolved")
