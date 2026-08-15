"""Tests for chain-pooled autocorrelation and integrated autocorrelation time."""

from __future__ import annotations

import pytest
import torch

from tpen.statistics.autocorrelation import (
    IntegratedAutocorrelation,
    integrated_autocorrelation_time,
    pooled_autocorrelation,
)


def _ar1_trajectory(
    phi: float,
    *,
    n_draws: int,
    n_walkers: int,
    burn_in: int,
    seed: int,
) -> torch.Tensor:
    """Generate independent Gaussian AR(1) chains after a fixed burn-in.

    Parameters
    ----------
    phi : float
        Autoregressive coefficient.
    n_draws : int
        Retained draws per chain.
    n_walkers : int
        Number of independent chains.
    burn_in : int
        Updates discarded before retention.
    seed : int
        Seed for the local generator.

    Returns
    -------
    torch.Tensor
        Samples with shape ``[n_draws, n_walkers]``.
    """

    generator = torch.Generator().manual_seed(seed)
    innovations = torch.randn(
        (burn_in + n_draws, n_walkers),
        generator=generator,
        dtype=torch.float64,
    )
    # Unit-variance initialization and innovation scaling put the process in
    # its stationary Gaussian law before the conservative burn-in begins.
    state = torch.randn(n_walkers, generator=generator, dtype=torch.float64)
    innovation_scale = (1.0 - phi**2) ** 0.5
    retained: list[torch.Tensor] = []
    for draw_index, innovation in enumerate(innovations):
        state = phi * state + innovation_scale * innovation
        if draw_index >= burn_in:
            retained.append(state.clone())
    return torch.stack(retained)


def _assert_unresolved(result: IntegratedAutocorrelation) -> str:
    """Assert that an unresolved result leaks no fabricated estimate.

    Parameters
    ----------
    result : IntegratedAutocorrelation
        Result expected to be unresolved.

    Returns
    -------
    str
        The non-empty unresolved reason for additional path-specific checks.
    """

    assert result.tau_int is None
    assert result.variance is None
    assert result.plateau_reached is False
    assert result.truncation_lag is None
    assert result.pair_count is None
    assert result.reason is not None
    assert result.reason.strip()
    return result.reason


def test_fft_matches_reference_for_multiple_chains() -> None:
    """Match the production FFT estimator to its direct-sum definition."""
    generator = torch.Generator().manual_seed(1701)
    # A non-power-of-two draw count forces the FFT path to zero-pad rather than
    # accidentally exercising an already convenient transform length.
    values = torch.randn((257, 5), generator=generator, dtype=torch.float64)

    actual = pooled_autocorrelation(values, method="fft")
    expected = pooled_autocorrelation(values, method="reference")

    # Direct summation accumulates O(n) float64 products at each lag, while the
    # FFT accumulates O(log n) butterfly error. This tolerance is comfortably
    # above those round-off bounds but far below a normalization discrepancy.
    torch.testing.assert_close(actual, expected, rtol=5.0e-13, atol=5.0e-13)


def test_non_degenerate_autocorrelation_starts_at_one() -> None:
    """Normalize every non-degenerate autocorrelation to one at lag zero."""
    generator = torch.Generator().manual_seed(1702)
    values = torch.randn((113, 4), generator=generator, dtype=torch.float64)

    rho = pooled_autocorrelation(values)

    assert rho[0].item() == pytest.approx(1.0, rel=0.0, abs=1.0e-14)


def test_known_ar1_matches_theoretical_integrated_autocorrelation_time() -> None:
    """Recover the analytic IAT of a stationary Gaussian AR(1) process."""
    phi = 0.8
    values = _ar1_trajectory(
        phi,
        n_draws=8192,
        n_walkers=16,
        burn_in=256,
        seed=1703,
    )

    result = integrated_autocorrelation_time(values)

    theoretical_tau = (1.0 + phi) / (1.0 - phi)
    assert theoretical_tau == pytest.approx(9.0)
    assert result.tau_int is not None
    # Here phi**30 is about 1.2e-3, so the useful IPS window is roughly 30
    # lags. With 8192 * 16 pooled observations, the standard window estimate
    # sqrt(2 * (2M + 1) / N) gives about 3.1% relative sampling error. A 15%
    # tolerance is therefore about five standard errors while still rejecting
    # a half-IAT convention or a materially biased truncation.
    assert result.tau_int == pytest.approx(theoretical_tau, rel=0.15)
    assert result.plateau_reached is True
    assert result.pair_count is not None
    assert result.truncation_lag == 2 * result.pair_count - 1


def test_iid_white_noise_has_unit_integrated_autocorrelation_time() -> None:
    """Recover unit IAT when every retained draw is independent."""
    generator = torch.Generator().manual_seed(1704)
    values = torch.randn((8192, 16), generator=generator, dtype=torch.float64)

    result = integrated_autocorrelation_time(values)

    assert result.tau_int is not None
    # Early-lag noise is O(1/sqrt(8192 * 16)) ~= 0.003. The wider 0.08 bound
    # accommodates the data-dependent IPS stopping point but catches an
    # off-by-one or factor-of-two error in the tau convention.
    assert result.tau_int == pytest.approx(1.0, abs=0.08)


def test_walker_boundaries_are_not_concatenated_into_false_correlation() -> None:
    """Keep independent walker offsets out of the serial-correlation estimate."""
    generator = torch.Generator().manual_seed(1705)
    a = 5.0 + torch.randn((4096, 1), generator=generator, dtype=torch.float64)
    b = -5.0 + torch.randn((4096, 1), generator=generator, dtype=torch.float64)

    separate = integrated_autocorrelation_time(torch.cat((a, b), dim=1))
    concatenated = integrated_autocorrelation_time(torch.cat((a, b), dim=0))

    assert separate.tau_int is not None
    assert separate.tau_int == pytest.approx(1.0, abs=0.20)
    assert concatenated.tau_int is not None
    # The artificial boundary creates a long +5 to -5 step. Requiring the
    # contrast, rather than only a near-one result, proves which axis the
    # estimator treats as time.
    assert concatenated.tau_int >= 10.0 * separate.tau_int


def test_too_few_draws_is_unresolved_without_a_numeric_estimate() -> None:
    """Withhold every estimate when a chain is shorter than the declared floor."""
    generator = torch.Generator().manual_seed(1706)
    values = torch.randn((7, 3), generator=generator, dtype=torch.float64)

    reason = _assert_unresolved(
        integrated_autocorrelation_time(values, min_draws_per_chain=8)
    )

    assert "insufficient draws" in reason


def test_constant_chains_are_unresolved_without_a_numeric_estimate() -> None:
    """Withhold every estimate when pooled variance is zero."""
    reason = _assert_unresolved(integrated_autocorrelation_time(torch.ones((64, 4))))

    assert "zero" in reason.lower()
    assert "variance" in reason.lower()


@pytest.mark.parametrize(
    "nonfinite",
    [pytest.param(float("nan"), id="nan"), pytest.param(float("inf"), id="inf")],
)
def test_nonfinite_trajectory_is_unresolved_without_a_numeric_estimate(
    nonfinite: float,
) -> None:
    """Withhold every estimate rather than dropping a non-finite draw."""
    generator = torch.Generator().manual_seed(1707)
    values = torch.randn((64, 4), generator=generator, dtype=torch.float64)
    values[11, 2] = nonfinite

    reason = _assert_unresolved(integrated_autocorrelation_time(values))

    assert "non-finite" in reason


def test_short_strongly_autocorrelated_chain_reports_no_plateau() -> None:
    """Report an unresolved result when every available Geyer pair is positive."""
    # This alternating trajectory is the phi -> -1 limit of an AR(1) chain.
    # Its biased rho_k is exactly (-1)^k * (n-k) / n, so every Geyer pair is
    # exactly 1/n > 0. The construction therefore reaches the no-plateau path
    # by invariant, without depending on a lucky random seed.
    values = torch.tensor([1.0, -1.0] * 8, dtype=torch.float64).reshape(-1, 1)
    rho = pooled_autocorrelation(values)

    assert rho[1].item() == pytest.approx(-15.0 / 16.0)
    reason = _assert_unresolved(integrated_autocorrelation_time(values))
    assert "plateau" in reason.lower()
    assert "short" in reason.lower()


def test_even_draw_count_uses_the_final_complete_geyer_pair() -> None:
    """Let the last odd lag terminate the positive sequence for an even length."""
    # For this centred 16-draw sequence, the unnormalised Geyer pair sums are
    # [13, 1, 1, 1, 1, 1, 1, -3] and gamma_0 is 32. Thus the final complete
    # pair (lags 14 and 15) is the first non-positive pair; omitting it would
    # falsely report no plateau even though all required values are available.
    values = torch.tensor(
        [3.0, *([-1.0, 1.0] * 7), -3.0],
        dtype=torch.float64,
    ).reshape(-1, 1)

    result = integrated_autocorrelation_time(values)

    assert result.tau_int == pytest.approx(3.0 / 16.0)
    assert result.plateau_reached is True
    assert result.pair_count == 7
    assert result.truncation_lag == 13


def test_integrated_autocorrelation_rejects_both_estimate_and_reason() -> None:
    """Reject a result that claims both resolution and failure."""
    with pytest.raises(ValueError, match="exactly one"):
        IntegratedAutocorrelation(
            tau_int=1.0,
            variance=1.0,
            plateau_reached=True,
            truncation_lag=1,
            pair_count=1,
            max_lag=7,
            reason="contradictory",
        )


def test_integrated_autocorrelation_rejects_neither_estimate_nor_reason() -> None:
    """Reject a result that records neither resolution nor failure."""
    with pytest.raises(ValueError, match="exactly one"):
        IntegratedAutocorrelation(
            tau_int=None,
            variance=None,
            plateau_reached=False,
            truncation_lag=None,
            pair_count=None,
            max_lag=7,
            reason=None,
        )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5,), id="one-dimensional"),
        pytest.param((0, 2), id="empty-draws"),
        pytest.param((3, 0), id="empty-walkers"),
    ],
)
@pytest.mark.parametrize(
    "estimator",
    [pooled_autocorrelation, integrated_autocorrelation_time],
)
def test_autocorrelation_estimators_reject_invalid_shapes(shape, estimator) -> None:
    """Require non-empty tensors with explicit draw and walker axes."""
    values = torch.empty(shape, dtype=torch.float64)

    with pytest.raises(ValueError):
        estimator(values)


@pytest.mark.parametrize(
    "estimator",
    [pooled_autocorrelation, integrated_autocorrelation_time],
)
def test_autocorrelation_estimators_reject_unknown_method(estimator) -> None:
    """Reject estimator names outside the public FFT/reference vocabulary."""
    values = torch.arange(16, dtype=torch.float64).reshape(8, 2)

    with pytest.raises(ValueError, match="method"):
        estimator(values, method="not-an-estimator")
