"""Estimator statistics shared by every baseline run adapter.

These helpers live here rather than inside an adapter because they describe
properties of a Monte Carlo series, not of any particular code's output format.
Both the FermiNet adapter (which reads ``train_stats.csv``) and the DeepQMC
adapter (which reads ``result.h5``) need all of them.

Three facts drove the contents of this module, each learned by getting a
published number wrong first.

**A naive standard error understates the uncertainty.** Successive VMC steps are
correlated, so ``sigma/sqrt(n)`` is too small. :func:`blocking_stderr` corrects
for that.

**The number of blocks at which blocking stops is NOT a decorrelation
measure.** It is the argmax of the standard error over blocking levels, and on a
nearly flat curve it is an argmax of noise. Reporting it as evidence of poor
decorrelation produced a false caveat on a headline result. Use
:func:`blocking_inflation`, which is the ratio the argmax was standing in for.

**A single tail average cannot detect non-convergence and looks stable when the
run is still descending.** :func:`sign_test` is the cheap check that can, and it
is distribution-free: it reads the sign pattern of consecutive window
differences rather than comparing error bars, because bar magnitude gives false
reassurance and false alarms in roughly equal measure.
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence

from experiments.baselines.errors import AdapterError

#: Smallest number of blocks that still supports a usable variance estimate.
#: Below this the standard error is itself so noisy that a larger value means
#: nothing, so blocking stops rather than reporting a spuriously wide bar.
MIN_BLOCKS = 32

#: Windows used by :func:`sign_test`. Eight rather than five because under
#: independence the probability of an all-same-sign run is ``2 * (1/2)**(n-1)``:
#: 12.5% at five windows, 1.56% at eight. At five, a converged run trips the
#: gate one time in eight, which is too noisy to hang an expensive verdict on.
SIGN_TEST_WINDOWS = 8

#: Fewest steps an estimator window may span. **This is an absolute count, not a
#: fraction, because the requirement is absolute**: the window has to be long
#: enough to average out the slow mode, which is a property of the series rather
#: than of the run length.
#:
#: A fraction alone fails in the small-run limit and did so measurably. At a 0.25
#: fraction a 20000-step run yields a 5000-step window, and emitting at that
#: window put Psiformer on helium at -2.1 microhartree and PauliNet at 100k at
#: -3.33 microhartree -- both BELOW the exact energy, which no variational
#: estimate can be. Widening to 10000 steps moved them to +1.16 and +15.29,
#: matching the independently computed reference column.
MIN_TAIL_STEPS = 10000


def select_tail(
    total_steps: int,
    fraction: float,
    *,
    min_steps: int = MIN_TAIL_STEPS,
    allow_below_floor: bool = False,
) -> int:
    """Return how many trailing steps the estimator window should span.

    The window is ``max(round(fraction * total_steps), min_steps)``: the
    fraction is the request and the floor only ever widens it. The floor is what
    makes this correct in the small-run limit; see :data:`MIN_TAIL_STEPS` for the
    measurement that motivated it.

    Parameters
    ----------
    total_steps : int
        Length of the full series.
    fraction : float
        Requested trailing fraction, in ``(0, 1]``.
    min_steps : int, optional
        Absolute floor on the window, applied only when the run is long enough
        to reach it.
    allow_below_floor : bool, optional
        Permit a window shorter than ``min_steps`` when the run itself is
        shorter. Off by default so that a too-short run is an explicit decision
        rather than a silent downgrade.

    Returns
    -------
    int
        Number of trailing steps to average, at least 2.

    Raises
    ------
    AdapterError
        If ``fraction`` is outside ``(0, 1]``; if ``total_steps`` is below two;
        or if the whole trace is shorter than ``min_steps`` and
        ``allow_below_floor`` is False. The message names both the run length and
        the floor, because "too short" is not actionable without them.

    Notes
    -----
    A run shorter than the floor cannot reach it. By default that raises: an
    estimate from a 500-step run is not comparable to one from 200000 steps, and
    a caller that genuinely wants it should say so via ``allow_below_floor``.

    Under ``allow_below_floor`` the fraction governs, so a quarter tail of a
    2000-step run is 500 steps. Returning the whole 2000 instead -- which is
    what clipping the widened window to ``total_steps`` used to do -- reports a
    window the caller never asked for and leaves ``fraction`` with no effect on
    the result, which is indistinguishable from honouring it.
    """

    if not 0.0 < fraction <= 1.0:
        raise AdapterError(f"tail fraction must be in (0, 1], got {fraction}")
    if total_steps < 2:
        raise AdapterError(f"need at least two steps to estimate, got {total_steps}")

    # The fraction is the request; the floor only widens it. Neither is clipped
    # to `total_steps` here, so a run too short to fill the floor stays a
    # visible case below instead of collapsing into the whole trace.
    requested = max(round(fraction * total_steps), 2)
    window = max(requested, min_steps)

    if window > total_steps:
        # This run cannot fill the floor at all.
        if not allow_below_floor:
            raise AdapterError(
                f"run of {total_steps} steps cannot fill the {min_steps}-step minimum "
                "estimator window; pass allow_below_floor to accept a shorter one and "
                "record that the estimate is provisional"
            )
        # Widening to `total_steps` would discard `fraction` entirely, so the
        # fraction governs the window instead. It is already within the run:
        # `round(fraction * total_steps) <= total_steps` for any fraction <= 1.
        window = requested

    return window


def blocking_stderr(
    values: Sequence[float],
    min_blocks: int = MIN_BLOCKS,
    *,
    allow_below_floor: bool = False,
) -> tuple[float, int | None]:
    """Return a correlation-corrected standard error by pair-average blocking.

    Implements Flyvbjerg-Petersen blocking: repeatedly average adjacent pairs,
    recording the standard error at each level. Correlated data shows the
    standard error rising with block size and then plateauing; the plateau is
    the honest bar.

    Parameters
    ----------
    values : sequence of float
        The series to estimate the mean's uncertainty for.
    min_blocks : int, optional
        Stop once fewer than this many blocks remain.
    allow_below_floor : bool, optional
        Permit a window too short for the ladder to run at all. Off by default:
        the only bar available for such a window is the uncorrected naive one,
        which understates, and these numbers are transcribed into receipts.

    Returns
    -------
    tuple of (float, int or None)
        The standard error, and the number of blocks it was computed from.

        The second element is returned for provenance only. **Do not read it as
        a decorrelation diagnostic** -- it is the block count at which the
        maximum happened to occur, which on a flat curve is determined by noise.
        Use :func:`blocking_inflation` for that.

        On a window shorter than ``min_blocks`` the ladder never runs. Under
        ``allow_below_floor`` the standard error is then the naive one --
        uncorrected for autocorrelation and therefore understating -- and the
        block count is ``None``, because no blocking level produced it. ``None``
        rather than an integer on purpose: ``0`` or ``len(values)`` stays
        arithmetically silent, so ``stderr * sqrt(blocks)`` or ``blocks > 1``
        downstream would yield a plausible number for a window that was never
        blocked, whereas ``None`` raises ``TypeError`` at the first arithmetic
        touch.

    Raises
    ------
    AdapterError
        If fewer than two values are supplied; if every value is identical, so
        the window has no measurable spread; or if the window is shorter than
        ``min_blocks`` and ``allow_below_floor`` is False.
    """

    data = [float(value) for value in values]
    if len(data) < 2:
        raise AdapterError("blocking needs at least two values")

    # No spread at all is not a spread of zero. Every ladder level would measure
    # 0.0, `stderr > best_stderr` would be false at each of them, and the 0.0
    # initialiser below would be returned as though it had been measured --
    # which records.py accepts, because it rejects only negative bars.
    spread = statistics.variance(data)
    if spread == 0.0:
        raise AdapterError(
            f"window of {len(data)} identical values has no measurable spread, so its "
            "standard error is unmeasurable rather than zero"
        )

    if len(data) < max(min_blocks, 2):
        # The ladder cannot run even once, so there is no correction to apply.
        # Refuse by default rather than answer with the naive bar: it
        # understates, and this path used to return the 0.0 initialiser, which
        # reads downstream as infinite precision.
        if not allow_below_floor:
            raise AdapterError(
                f"window of {len(data)} values cannot fill the {min_blocks}-block minimum "
                "for blocking; pass allow_below_floor to accept the uncorrected naive "
                "standard error and record that it understates"
            )
        # None, not a block count: no blocking level produced this value.
        return math.sqrt(spread / len(data)), None

    # The ladder below is unchanged. It is now entered only when it will run at
    # least once, and `spread > 0` there, so a measured level always overwrites
    # `best_stderr` and the 0.0 initialiser can no longer escape.
    best_stderr = 0.0
    best_blocks = len(data)
    while len(data) >= max(min_blocks, 2):
        stderr = math.sqrt(statistics.variance(data) / len(data))
        if stderr > best_stderr:
            best_stderr, best_blocks = stderr, len(data)
        # Pair-average into the next blocking level, dropping a trailing odd
        # sample rather than pairing it with nothing.
        data = [(data[i] + data[i + 1]) / 2.0 for i in range(0, len(data) - 1, 2)]

    return best_stderr, best_blocks


def blocking_inflation(
    values: Sequence[float],
    min_blocks: int = MIN_BLOCKS,
    *,
    allow_below_floor: bool = False,
) -> float | None:
    """Return how much blocking widens the bar relative to the naive estimate.

    This is the autocorrelation diagnostic. A ratio near 1 means the series is
    effectively uncorrelated and the naive standard error is nearly right; a
    large ratio means successive samples carry much less independent information
    than their count suggests.

    Parameters
    ----------
    values : sequence of float
        The series, normally an already-selected tail.
    min_blocks : int, optional
        Passed through to :func:`blocking_stderr`.
    allow_below_floor : bool, optional
        Passed through to :func:`blocking_stderr`. With it set, a window too
        short to block returns ``None`` instead of raising.

    Returns
    -------
    float or None
        ``blocked_stderr / naive_stderr``, at least 1.0 by construction since
        blocking takes the maximum over levels and level one is the naive value.

        ``None`` when the window was too short for the ladder to run and
        ``allow_below_floor`` is set. There is no blocked-to-naive comparison to
        report in that case, and the arithmetic answer would be exactly 1.00 --
        which reads as "blocking found no autocorrelation here", the most
        reassuring value this diagnostic can take, for a window never blocked.

    Raises
    ------
    AdapterError
        If fewer than two values are supplied; if the series has zero variance
        so the ratio is undefined; or if the window is shorter than
        ``min_blocks`` and ``allow_below_floor`` is False.
    """

    data = [float(value) for value in values]
    if len(data) < 2:
        raise AdapterError("blocking inflation needs at least two values")

    naive = math.sqrt(statistics.variance(data) / len(data))
    if naive == 0.0:
        raise AdapterError("blocking inflation is undefined for a constant series")

    blocked, blocks = blocking_stderr(
        data, min_blocks=min_blocks, allow_below_floor=allow_below_floor
    )
    if blocks is None:
        return None
    return blocked / naive


def window_means(values: Sequence[float], windows: int = SIGN_TEST_WINDOWS) -> list[float]:
    """Split a series into equal windows and return each window's mean.

    Trailing samples that do not fill a whole window are dropped, so every
    window carries the same weight.

    Raises
    ------
    AdapterError
        If ``windows`` is below two, or the series cannot fill that many
        windows with at least one sample each.
    """

    if windows < 2:
        raise AdapterError(f"need at least two windows, got {windows}")

    data = [float(value) for value in values]
    width = len(data) // windows
    if width < 1:
        raise AdapterError(f"series of {len(data)} cannot fill {windows} windows")

    return [statistics.fmean(data[i * width : (i + 1) * width]) for i in range(windows)]


def sign_test(values: Sequence[float], windows: int = SIGN_TEST_WINDOWS) -> tuple[str, bool]:
    """Test a series for monotone drift via the sign pattern of its windows.

    Returns the sign string of the consecutive window differences and whether
    that pattern is monotone. A monotone run means the series is still moving in
    one direction -- for a training loss, that it has not converged. A mixed
    pattern means noise, however large the excursions are.

    **Why signs and not error bars.** Bar magnitude fails in both directions. A
    run whose windows drifted 37 microhartree monotonically with 10 microhartree
    bars had every adjacent pair overlapping, so an overlap test called it flat
    while its cumulative drift was 3.7 bars. A different run scattered 104
    microhartree against an 18 microhartree bar and was pure noise, because the
    movement was non-monotone. Only the sign pattern separates those.

    Parameters
    ----------
    values : sequence of float
        The series to test, normally a trailing segment long enough that each
        window exceeds the autocorrelation time of the series being windowed.
    windows : int, optional
        Number of equal windows. Defaults to :data:`SIGN_TEST_WINDOWS`.

    Returns
    -------
    tuple of (str, bool)
        The sign string, one character per consecutive difference, and True if
        the pattern is monotone in either direction.

    Notes
    -----
    The false-alarm rate quoted for this test assumes **independent window
    means**. If a window is narrower than the autocorrelation time of the
    series, consecutive means are positively correlated, same-sign runs become
    more likely under the null, and the nominal rate is optimistic by an
    unbounded factor. For a training loss that correlation time is a property of
    the optimizer trajectory and is generally much longer than the sampler's,
    so the sampler's value must not be substituted for it.
    """

    means = window_means(values, windows=windows)
    diffs = [means[i + 1] - means[i] for i in range(len(means) - 1)]
    signs = "".join("-" if diff < 0 else "+" for diff in diffs)
    monotone = signs == "-" * len(diffs) or signs == "+" * len(diffs)
    return signs, monotone


__all__ = [
    "MIN_BLOCKS",
    "MIN_TAIL_STEPS",
    "SIGN_TEST_WINDOWS",
    "blocking_inflation",
    "blocking_stderr",
    "select_tail",
    "sign_test",
    "window_means",
]
