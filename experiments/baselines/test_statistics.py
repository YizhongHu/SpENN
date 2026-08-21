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
    MIN_BLOCKS,
    blocking_inflation,
    blocking_stderr,
    select_tail,
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


def _reference_ladder(series: list[float], min_blocks: int = MIN_BLOCKS) -> tuple[float, int]:
    """The blocking ladder written out level by level, independently.

    Deliberately slow and literal: measure a level, then pair-average into the
    next, stopping once fewer than ``min_blocks`` blocks remain. Its purpose is
    to pin WHICH levels the real implementation visits, so that guards added in
    front of the ladder cannot quietly change a long-window bar.
    """

    data = list(series)
    best_stderr, best_blocks = 0.0, len(data)
    while len(data) >= max(min_blocks, 2):
        stderr = _naive_stderr(data)
        if stderr > best_stderr:
            best_stderr, best_blocks = stderr, len(data)
        data = [(data[i] + data[i + 1]) / 2.0 for i in range(0, len(data) - 1, 2)]
    return best_stderr, best_blocks


def test_select_tail_never_exceeds_the_run_length() -> None:
    """The window is a slice of the series, so it cannot be longer than it.

    A window longer than the run is both a lie about how many steps were
    averaged AND, when it comes from widening to the floor, a silent bypass of
    the short-run refusal: the window is no longer below `min_steps`.
    """

    assert select_tail(20000, 0.25) == 10000
    assert select_tail(12000, 0.9) == 10800
    assert select_tail(10000, 1.0) == 10000
    # Even asking for the whole trace cannot exceed it.
    assert select_tail(11000, 1.0) == 11000


def test_select_tail_applies_the_floor_and_then_the_fraction() -> None:
    """Floor wins below it, fraction wins above it."""

    assert select_tail(20000, 0.25) == 10000       # floor beats 5000
    assert select_tail(200000, 0.25) == 50000      # fraction beats the floor
    assert select_tail(40000, 0.25) == 10000       # exactly at the floor


def test_select_tail_refuses_a_run_below_the_floor() -> None:
    with pytest.raises(AdapterError, match="run of 500 steps cannot fill the 10000-step"):
        select_tail(500, 0.25)


def test_select_tail_honours_an_explicit_short_window() -> None:
    """A caller may take a short window deliberately; that is not the same as
    getting one by accident."""

    assert select_tail(500, 1.0, allow_below_floor=True) == 500


def test_select_tail_honours_the_fraction_when_the_floor_is_unreachable() -> None:
    """A fraction that changes nothing is a fraction that was not applied.

    The window used to be the floor-widened request clipped to the run, so any
    run shorter than the floor got the whole trace back, `fraction` had no
    effect on the result, and nothing in the return said so.
    """

    assert select_tail(2000, 0.25, min_steps=10000, allow_below_floor=True) == 500
    # The default floor is the same case, reached without naming the floor.
    assert select_tail(2000, 0.25, allow_below_floor=True) == 500
    # Asking for the whole trace still gets the whole trace.
    assert select_tail(2000, 1.0, min_steps=10000, allow_below_floor=True) == 2000


def test_select_tail_below_the_floor_distinguishes_two_fractions() -> None:
    """Two fractions of one short run must give two different windows.

    Written as a comparison on purpose: it fails for the whole family of bugs
    that return the run length whatever the fraction, not just for one slip.
    """

    quarter = select_tail(2000, 0.25, allow_below_floor=True)
    half = select_tail(2000, 0.5, allow_below_floor=True)

    assert quarter != half
    assert half == 2 * quarter


def test_select_tail_below_the_floor_returns_two_steps_at_minimum() -> None:
    """A variance needs two samples, however small the fraction of a tiny run."""

    assert select_tail(4, 0.25, allow_below_floor=True) == 2


def test_select_tail_rejects_a_bad_fraction() -> None:
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(AdapterError, match="tail fraction must be in"):
            select_tail(20000, bad)


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


@pytest.mark.parametrize("count", [2, 31])
def test_blocking_refuses_a_window_below_the_block_floor(count: int) -> None:
    """A window too short to block must not answer with an unmeasured number.

    `best_stderr` started at 0.0 and the ladder never ran below `min_blocks`, so
    a short window returned 0.0. records.py rejects only negative bars, so that
    0.0 validated and satisfied the "an energy needs a bar" check: the run
    published as though it had infinite precision.
    """

    assert count < MIN_BLOCKS
    with pytest.raises(AdapterError, match="cannot fill the 32-block"):
        blocking_stderr(_ar1(0.5, count, seed=11))


@pytest.mark.parametrize("count", [2, 31])
def test_blocking_below_the_floor_is_naive_with_no_block_count(count: int) -> None:
    """Opting in gets the naive bar and `None` where a block count would go.

    `None` rather than 0 or `len(values)`: an integer keeps working in
    arithmetic, so `stderr * sqrt(blocks)` or `blocks / total` downstream would
    produce a plausible number for a window that was never blocked.
    """

    series = _ar1(0.5, count, seed=11)
    blocked, blocks = blocking_stderr(series, allow_below_floor=True)

    assert blocked == pytest.approx(_naive_stderr(series))
    assert blocks is None
    # The marker has to be out of band, not merely unusual.
    with pytest.raises(TypeError):
        math.sqrt(blocks)


@pytest.mark.parametrize("allow_below_floor", [False, True])
def test_blocking_refuses_a_window_with_no_measurable_spread(allow_below_floor: bool) -> None:
    """Identical values are not a spread of zero, whatever the caller allows.

    The second route to the same zero bar: at or above `min_blocks` the ladder
    does run, but every level measures 0.0 and `stderr > best_stderr` is false
    at 0.0, so the unmeasured initialiser was returned anyway.
    """

    with pytest.raises(AdapterError, match="no measurable spread"):
        blocking_stderr([2.5] * 4096, allow_below_floor=allow_below_floor)


@pytest.mark.parametrize("count", [*range(MIN_BLOCKS, MIN_BLOCKS + 34), 256, 1000, 4096])
def test_blocking_on_the_laddered_path_always_reports_a_block_count(count: int) -> None:
    """`None` must be reachable only below the floor, never from the ladder.

    Otherwise the marker is a claim about the code rather than a property of it:
    a `None` escaping the ladder would reach a caller that has been told `None`
    means "too short to block at all".
    """

    blocked, blocks = blocking_stderr(_ar1(0.7, count, seed=count))

    assert blocked > 0.0
    assert blocks is not None
    assert MIN_BLOCKS <= blocks <= count


def test_blocking_visits_the_same_levels_as_a_reference_ladder() -> None:
    """The short-window guards sit in front of the ladder and must not alter it.

    A guard that changed which levels a long window visits would move published
    bars, because a coarser level is noisier and can take the argmax.
    """

    series = _ar1(0.9, 8192, seed=17)
    blocked, blocks = blocking_stderr(series)
    reference_stderr, reference_blocks = _reference_ladder(series)

    assert blocks == reference_blocks
    assert blocked == pytest.approx(reference_stderr)


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


def test_inflation_refuses_a_window_below_the_block_floor() -> None:
    """No ladder ran, so there is no blocked-to-naive ratio to report."""

    with pytest.raises(AdapterError, match="cannot fill the 32-block"):
        blocking_inflation(_ar1(0.9, 20, seed=13))


def test_inflation_below_the_floor_is_none_not_one() -> None:
    """1.00x reads as "blocking found no autocorrelation"; nothing was measured.

    Below the floor the numerator IS the denominator, so the arithmetic answer
    is exactly the most reassuring value this diagnostic can take -- and note 12
    of this program requires the ratio to be transcribed beside every bar, so a
    silent 1.00 would be copied into a receipt as evidence.
    """

    assert blocking_inflation(_ar1(0.9, 20, seed=13), allow_below_floor=True) is None


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
