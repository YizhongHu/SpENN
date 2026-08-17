"""Tests for chain-pooled autocorrelation and integrated autocorrelation time."""

from __future__ import annotations

import math

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


def _two_timescale_trajectory(
    fast_phi: float,
    slow_phi: float,
    slow_weight: float,
    *,
    n_draws: int,
    n_walkers: int,
    burn_in: int,
    seed: int,
) -> torch.Tensor:
    """Superpose two independent unit-variance AR(1) processes.

    The combination ``sqrt(1 - w) * fast + sqrt(w) * slow`` is again
    unit-variance, so its autocorrelation is the weighted sum of the two
    exponentials and the integrated autocorrelation time is

    ``tau_int = 1 + 2 * [ (1 - w) * pf / (1 - pf) + w * ps / (1 - ps) ]``.

    A single AR(1) has exactly one exponential timescale, so any window wider
    than that decay reproduces its ``tau_int``. Two separated timescales are
    what make an early truncation visible at all.

    Parameters
    ----------
    fast_phi, slow_phi : float
        Autoregressive coefficients of the two components.
    slow_weight : float
        Fraction ``w`` of the total variance carried by the slow component.
    n_draws : int
        Retained draws per chain.
    n_walkers : int
        Number of independent chains.
    burn_in : int
        Updates discarded before retention, sized on the slow component.
    seed : int
        Seed of the fast component; the slow component uses ``seed + 10000`` so
        the two are drawn from disjoint streams.

    Returns
    -------
    torch.Tensor
        Samples with shape ``[n_draws, n_walkers]``.
    """

    fast = _ar1_trajectory(
        fast_phi, n_draws=n_draws, n_walkers=n_walkers, burn_in=burn_in, seed=seed
    )
    slow = _ar1_trajectory(
        slow_phi, n_draws=n_draws, n_walkers=n_walkers, burn_in=burn_in, seed=seed + 10_000
    )
    return math.sqrt(1.0 - slow_weight) * fast + math.sqrt(slow_weight) * slow


_HETEROGENEOUS_SCALES = (1.0, 3.0, 10.0, 30.0)
"""Per-chain amplitudes spanning a 900-fold range of chain variance."""


def _scale_heterogeneous_chains(
    phi: float,
    *,
    scales: tuple[float, ...],
    n_draws: int,
    burn_in: int,
    seed: int,
) -> torch.Tensor:
    """Return AR(1) chains sharing one ``phi`` but spanning a wide variance range.

    Parameters
    ----------
    phi : float
        Autoregressive coefficient shared by every chain.
    scales : tuple of float
        Per-chain amplitude; chain variance scales as the square.
    n_draws : int
        Retained draws per chain.
    burn_in : int
        Updates discarded before retention.
    seed : int
        Seed for the underlying unit-variance chains.

    Returns
    -------
    torch.Tensor
        Samples with shape ``[n_draws, len(scales)]``.
    """

    base = _ar1_trajectory(
        phi, n_draws=n_draws, n_walkers=len(scales), burn_in=burn_in, seed=seed
    )
    return base * torch.tensor(scales, dtype=torch.float64)


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


def test_pooled_autocorrelation_survives_a_nine_hundred_fold_variance_spread() -> None:
    """Recover the shared decay when chain variances span three orders of magnitude.

    ``pooled_autocorrelation`` averages autocovariances and divides by the
    *pooled* lag-zero term, so a chain contributes in proportion to its own
    variance -- deliberate, so one stuck walker cannot dominate. With
    amplitudes [1, 3, 10, 30] the largest chain carries 89% of the pooling
    weight and the effective chain count is 1.24, not 4, which is why this test
    pins the correlations and leaves ``tau_int`` to the theoretical fixture.

    Note that per-chain scaling is NOT an exact invariance even in exact
    arithmetic: it reweights the chains, and
    ``mean_c(s_c^2 g_c(k)) / mean_c(s_c^2 g_c(0))`` does not reduce to
    ``mean_c(g_c(k)) / mean_c(g_c(0))``. Only a global factor is invariant, and
    that is asserted separately.
    """
    phi = 0.8
    values = _scale_heterogeneous_chains(
        phi,
        scales=_HETEROGENEOUS_SCALES,
        n_draws=8192,
        burn_in=256,
        # Median seed of a 24-seed calibration ensemble.
        seed=1914,
    )

    rho = pooled_autocorrelation(values)

    assert rho[0].item() == pytest.approx(1.0, rel=0.0, abs=1.0e-14)
    # The tolerance is absolute because the compared quantity is a correlation
    # bounded by one and does NOT sweep magnitude across this test -- the
    # 900-fold spread is in the input. Measured over 24 seeds, rho[1] has sd
    # 0.0064 and rho[2] has sd 0.0113, so these bounds are 7.8 and 5.3 standard
    # errors. Normalising by any single chain's lag-zero term instead of the
    # pooled one moves rho[1] to 202 (first chain) or 0.224 (largest chain).
    assert rho[1].item() == pytest.approx(phi, abs=0.05)
    assert rho[2].item() == pytest.approx(phi**2, abs=0.06)
    # WHAT THIS DOES NOT CATCH, measured rather than assumed: replacing the
    # variance-proportional pooling with equal weighting, or with amplitude
    # weighting, leaves every test in the suite passing. Chains that share one
    # phi cannot separate the weightings at all, since each returns phi**k
    # regardless. Amplitude heterogeneity is therefore the wrong axis for that
    # property; it needs chains with differing correlation times.


@pytest.mark.parametrize(
    "factor",
    [
        pytest.param(2.0**-10, id="2**-10"),
        pytest.param(8.0, id="8"),
        pytest.param(2.0**20, id="2**20"),
    ],
)
def test_exactly_representable_global_rescaling_is_bitwise_invariant(factor: float) -> None:
    """Leave the pooled autocorrelation bit-for-bit unchanged under a power of two.

    Multiplying by a power of two is a pure exponent shift, exact for every
    element and for every partial sum, so numerator and denominator are exactly
    scaled copies and the ratio is unchanged in the last bit. That makes this
    the strictest available probe for scale-dependent logic: any coupling
    between the estimator and the magnitude of its input breaks exact
    invariance even for an exactly-representable factor.

    This is a different claim from the inexact-factor test below, which bounds
    rounding rather than excluding structure. The gap between them is a
    property of binary floating point, not a weakness of the estimator, and
    neither test subsumes the other.
    """
    values = _scale_heterogeneous_chains(
        0.8, scales=_HETEROGENEOUS_SCALES, n_draws=1024, burn_in=256, seed=1914
    )

    rescaled = pooled_autocorrelation(values * factor)

    # Measured residual across the calibration ensemble was exactly zero, so
    # this asserts equality rather than a tolerance.
    assert torch.equal(rescaled, pooled_autocorrelation(values))


@pytest.mark.parametrize(
    "factor",
    [
        pytest.param(3.7, id="3.7"),
        pytest.param(1.0 / 3.0, id="1/3"),
        pytest.param(1.0e-3, id="1e-3"),
    ],
)
def test_inexact_global_rescaling_perturbs_only_at_ulp_scale(factor: float) -> None:
    """Bound the rounding residual of a global factor that is not a power of two.

    A factor that is not exactly representable rounds every scaled element
    independently, and those errors do not cancel between numerator and
    denominator because the two are different sums over differently-rounded
    terms. The ratio therefore moves at ulp scale rather than not at all.

    The bound is stated in ulp of *one*, not of each ``rho[k]``: the residual is
    inherited from the lag-zero normalisation and is flat in lag, so it stays
    near 1e-16 even where ``rho[k]`` has decayed to 1e-11. A per-lag
    ``ulp(rho[k])`` bound is unsatisfiable in the tail for exactly that reason
    -- measured, it would demand agreement 650000 times tighter than float64
    can express there.
    """
    values = _scale_heterogeneous_chains(
        0.8, scales=_HETEROGENEOUS_SCALES, n_draws=1024, burn_in=256, seed=1914
    )

    rescaled = pooled_autocorrelation(values * factor)
    residual = (rescaled - pooled_autocorrelation(values)).abs()

    # Worst residual measured across nine factors and eight seeds was 1.5 ulp
    # of one; this bound is 8 ulp, and rejects a scale coupling that would show
    # up as a relative shift many orders of magnitude larger.
    assert residual.max().item() <= 8.0 * math.ulp(1.0)


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


def test_two_timescale_plateau_is_not_truncated_at_the_fast_decay() -> None:
    """Sum the slow tail of a two-timescale process, not just the fast decay.

    Every other fixture here drives a single AR(1), which has one exponential
    timescale, so none of them can tell a correct truncation from one that
    merely stops early. Here the slow component carries 10% of the variance but
    78% of the excess correlation time, so an estimator that stops after the
    fast decay reports a number that is not merely imprecise but wrong by more
    than half.
    """
    fast_phi, slow_phi, slow_weight = 0.5, 0.97, 0.1
    values = _two_timescale_trajectory(
        fast_phi,
        slow_phi,
        slow_weight,
        n_draws=8192,
        n_walkers=16,
        # Thirty slow timescales, so the retained draws are stationary in the
        # component that dominates tau_int rather than only in the fast one.
        burn_in=1024,
        # Median seed of a 24-seed calibration ensemble, chosen deliberately:
        # a seed picked for passing would hide how much margin the bounds have.
        seed=1819,
    )

    result = integrated_autocorrelation_time(values)

    theoretical_tau = 1.0 + 2.0 * (
        (1.0 - slow_weight) * fast_phi / (1.0 - fast_phi)
        + slow_weight * slow_phi / (1.0 - slow_phi)
    )
    assert theoretical_tau == pytest.approx(9.266667, abs=1.0e-6)
    assert result.tau_int is not None
    # The band is ASYMMETRIC because the estimator's error here is signed:
    # initial-positive-sequence truncation and the monotone envelope can only
    # remove tail mass, never manufacture it. Measured over 24 seeds, tau_int
    # has mean 8.60 and sd 0.52, i.e. a 7% low bias with four contributions --
    # the forfeited tail beyond the stop (2.4%), the per-chain mean-subtraction
    # bias of the biased autocovariance (2.7%), the monotone envelope itself
    # (1.8%), and the triangular taper (0.3%). The lower bound sits 4.1 sd
    # below that mean while still rejecting an eight-lag window, which reports
    # at most 4.30 on this fixture. The upper bound is looser than symmetry
    # would suggest, and for a mechanism rather than an order statistic: the
    # distribution is right-skewed because a longer stop both accumulates more
    # noise and forfeits less tail, so the two effects push tau_int up
    # together. COUNTED over the 24 calibration seeds:
    # corr(truncation_lag, tau_int) = +0.385; the long-stop group (lag >= 120,
    # n=9: seeds 1803 1804 1805 1806 1807 1808 1811 1816 1821) has mean tau
    # 8.8501, against 8.3270 for the short-stop group (lag <= 100, n=9: seeds
    # 1801 1802 1809 1810 1813 1814 1815 1819 1823). No seed sits at lag 100, so
    # the inclusion rule cannot move one.
    # THE MARGIN ITSELF IS MODELLED, not counted, and the sample cannot settle
    # it: the window formula sqrt(2 * (2K + 1) / N) * tau puts this bound ~3.24
    # sd above the long-stop mean. The measured alternative does NOT adjudicate
    # that, because at n=9 the sample sd 0.7317 carries a standard error of
    # s/sqrt(2(n-1)) = 0.1829 -- 25% relative -- so the implied margin spans 2.48
    # to 4.14 and the modelled value sits inside. Model and measurement agree to
    # within the precision available; neither supports a claim that one is wider.
    # Earlier revisions of this comment asserted +0.393, 8.9077, 8.3615, 0.6457
    # and "wider than the model". Those came from a HAND-TRANSCRIBED copy of the
    # calibration table that dropped one row's lag and padded the end, silently
    # mispairing 10 of 24 rows -- a length check passed because the pad restored
    # the length. Read such columns from the retained log programmatically.
    assert 0.70 * theoretical_tau <= result.tau_int <= 1.20 * theoretical_tau
    assert result.plateau_reached is True
    # The more direct statement of the same property, and the sharper
    # instrument: an eight-lag window truncates at lag 7. The measured stop lag
    # has median 107 and minimum 75 across the calibration ensemble, so this
    # bound is far below anything correct code produces and far above the
    # mutant. It tests "the window did not stop early" without routing that
    # claim through a noisy derived scalar.
    assert result.truncation_lag is not None
    assert result.truncation_lag >= 30
    # DETECTION REACH, stated because a coverage test that reports only its
    # successes is as misleading as one that checks only the side it expects to
    # fail. A truncation at K lags reports 1 + 2 * sum_{k<=K} rho(k) here, so
    # the band alone catches K <= 27 (tau 6.4254) and passes K >= 28 (tau
    # 6.5106), while the truncation_lag bound catches any cap of 15 pairs or
    # fewer, i.e. K <= 29. Total reach is K <= 29 at these bounds, and
    # truncation_lag is the lever worth reaching for -- but NOT because the band
    # is immovable. Tightening the lower fraction f DOES raise the band's cap:
    # f=0.75 reaches K <= 33 with 3.17 sd of margin, f=0.80 reaches 41 at
    # 2.28 sd, f=0.90 reaches 63 at 0.50 sd. The band becomes useless on
    # FALSE-ALARM GROUNDS rather than arithmetic ones, and two thresholds answer
    # two different questions: reaching the healthy-stop regime K >= 75 needs
    # f > 0.92894, whose lower bound 8.6082 sits 0.012 sd ABOVE the observed mean
    # of 8.6018 and is exceeded by 16 of the 24 calibration seeds; merely closing
    # the [30, 75) gap needs f > 0.92674, lower bound 8.5878, which 15 of 24
    # exceed. Either way a clear majority of correct runs -- roughly two thirds
    # of the measured seeds -- would fail, so the gap is closable only through
    # truncation_lag. Those counts are EMPIRICAL, and deliberately so: earlier
    # revisions of this comment quoted 50.5% and 48.9% from a normal tail
    # probability, which understates the rejection rate by about 15 points
    # because the tau_int distribution is right-skewed (mean 8.6018, median
    # 8.5033) -- the same skew this comment invokes twelve lines above to justify
    # the asymmetric bound. Count where you can, model where you must, and label
    # which you did.
    # THE LIMIT THAT FOLLOWS, measured: a defect truncating at 50 lags reports
    # 7.8565 and passes both assertions, as does one at 75 (8.6082), while the
    # smallest healthy stop over 24 seeds was 75 and the median 107. So
    # K in [30, 75) is UNDETECTED. This catches gross early truncation, not
    # moderate early truncation. Verified by mutation: a 12-pair cap (23 lags)
    # fails this test and nothing else in the suite, and a 16-pair cap (31 lags)
    # fails nothing at all.


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
