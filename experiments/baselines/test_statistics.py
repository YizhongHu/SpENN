"""Tests for the shared estimator statistics.

Each test here corresponds to a specific way this program has previously
reported a wrong number, so the docstrings name the failure rather than
restating the code.
"""

from __future__ import annotations

import math
import random

import pytest

from experiments.baselines.errors import AdapterError
from experiments.baselines.statistics import (
    blocking_inflation,
    blocking_stderr,
    sign_test,
    window_means,
)


def _ar1(rho: float, count: int, seed: int) -> list[float]:
    """Return an AR(1) series, positively correlated for ``rho > 0``."""

    rng = random.Random(seed)
    value, series = 0.0, []
    for _ in range(count):
        value = rho * value + rng.gauss(0.0, 1.0)
        series.append(value)
    return series


def _naive_stderr(series: list[float]) -> float:
    mean = sum(series) / len(series)
    variance = sum((x - mean) ** 2 for x in series) / (len(series) - 1)
    return math.sqrt(variance / len(series))


def test_blocking_exceeds_naive_stderr_on_correlated_data() -> None:
    """Blocking must inflate the bar on autocorrelated input.

    This is the property that makes the error bar honest: an AR(1) series has
    fewer independent samples than points, and the naive standard error does not
    know that.
    """

    series = _ar1(0.9, 8192, seed=20260813)
    blocked, _ = blocking_stderr(series)
    assert blocked > _naive_stderr(series) * 1.5


def test_blocking_matches_naive_on_independent_data() -> None:
    """On uncorrelated input the two estimates should be close."""

    rng = random.Random(11)
    series = [rng.gauss(0.0, 1.0) for _ in range(8192)]
    blocked, _ = blocking_stderr(series)
    assert 0.6 / math.sqrt(len(series)) < blocked < 1.8 / math.sqrt(len(series))


def test_blocking_rejects_single_sample() -> None:
    with pytest.raises(AdapterError, match="at least two values"):
        blocking_stderr([1.0])


def test_inflation_is_large_when_correlated_and_near_one_when_not() -> None:
    """The inflation ratio is the real autocorrelation diagnostic.

    The block COUNT at which blocking stops is not: it is the argmax of the
    standard error over blocking levels, so on a flat curve it is set by noise.
    Reporting that count as evidence of poor decorrelation produced a false
    caveat on a headline result in this program; the ratio is what it was
    standing in for.
    """

    correlated = blocking_inflation(_ar1(0.9, 8192, seed=7))
    independent_rng = random.Random(7)
    independent = blocking_inflation([independent_rng.gauss(0.0, 1.0) for _ in range(8192)])

    assert correlated > 2.0
    assert independent < 1.3
    assert correlated > independent


def test_inflation_is_at_least_one() -> None:
    """Blocking takes the maximum over levels, and level one is the naive value."""

    assert blocking_inflation(_ar1(0.5, 4096, seed=3)) >= 1.0


def test_inflation_rejects_a_constant_series() -> None:
    """A zero-variance series has no defined ratio and must not return 0 or 1."""

    with pytest.raises(AdapterError, match="constant series"):
        blocking_inflation([2.5] * 4096)


def test_sign_test_detects_monotone_drift_that_bars_would_miss() -> None:
    """A steadily descending series is caught even when adjacent windows overlap.

    Built to match the real case: eight windows drifting down by well under one
    error bar per step, so an adjacent-overlap test calls it flat while the
    cumulative movement is several bars. The sign pattern is what separates the
    two.
    """

    rng = random.Random(99)
    series = [(-0.000001 * i) + rng.gauss(0.0, 0.00002) for i in range(80000)]
    signs, monotone = sign_test(series)
    assert signs == "-------"
    assert monotone is True


def test_sign_test_calls_large_non_monotone_scatter_noise() -> None:
    """Excursions far larger than the bar are still noise if non-monotone.

    The counter-case to the test above, and the reason bar magnitude is the
    wrong criterion: it fails in both directions.
    """

    rng = random.Random(4)
    series = [rng.gauss(0.0, 1.0) for _ in range(80000)]
    signs, monotone = sign_test(series)
    assert monotone is False
    assert set(signs) == {"+", "-"}


def test_sign_test_default_is_eight_windows() -> None:
    """Seven differences, not four: 1.56% false-alarm rate rather than 12.5%."""

    signs, _ = sign_test(list(range(8000)))
    assert len(signs) == 7


def test_sign_test_rejects_a_series_too_short_to_window() -> None:
    """Too short must fail loudly, not silently shrink the windows to fit."""

    with pytest.raises(AdapterError, match="cannot fill"):
        sign_test([1.0, 2.0, 3.0])


def test_window_means_drops_the_partial_trailing_window() -> None:
    """Every window carries equal weight, so a short remainder is discarded."""

    means = window_means(list(range(101)), windows=10)
    assert len(means) == 10
    # Width 10, so the last window covers 90..99 and index 100 is dropped.
    assert means[-1] == pytest.approx(94.5)


def test_window_means_requires_at_least_two_windows() -> None:
    with pytest.raises(AdapterError, match="at least two windows"):
        window_means([1.0, 2.0], windows=1)
