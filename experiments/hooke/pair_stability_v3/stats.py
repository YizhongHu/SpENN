"""Study-local numeric helpers for staged reductions."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Sequence

DEFAULT_BAR_QUANTILE_RANGE = (0.05, 0.85)


def as_float(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None``."""

    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_bool(value: Any) -> bool | None:
    """Return ``value`` as a bool when it is represented explicitly."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def finite_values(values: Iterable[Any]) -> list[float]:
    """Return finite floats parsed from ``values``."""

    return [parsed for parsed in (as_float(value) for value in values) if parsed is not None]


def mean(values: Iterable[Any]) -> float | None:
    """Return the finite-value mean, or ``None`` for no finite values."""

    clean = finite_values(values)
    return statistics.fmean(clean) if clean else None


def median(values: Iterable[Any]) -> float | None:
    """Return the finite-value median, or ``None`` for no finite values."""

    clean = finite_values(values)
    return statistics.median(clean) if clean else None


def variance(values: Iterable[Any]) -> float | None:
    """Return the finite-value sample variance."""

    clean = finite_values(values)
    if len(clean) < 2:
        return 0.0 if clean else None
    return statistics.variance(clean)


def quantile(values: Iterable[Any], q: float) -> float | None:
    """Return the finite-value linear-interpolated quantile."""

    clean = sorted(finite_values(values))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return clean[int(position)]
    weight = position - low
    return clean[low] * (1.0 - weight) + clean[high] * weight


def finite_sum(values: Iterable[Any]) -> float | None:
    """Return the finite-value sum, or ``None`` for no finite values."""

    clean = finite_values(values)
    return math.fsum(clean) if clean else None


def finite_max(values: Iterable[Any]) -> float | None:
    """Return the finite-value maximum, or ``None`` for no finite values."""

    clean = finite_values(values)
    return max(clean) if clean else None


def format_number(value: float | None) -> str:
    """Return the compact numeric string used by study CSV artifacts."""

    if value is None:
        return ""
    return f"{value:.12g}"


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float | None:
    """Return a weighted empirical quantile for histogram-like bins."""

    pairs = sorted(
        (value, weight)
        for value, weight in zip(values, weights, strict=True)
        if math.isfinite(value) and weight > 0.0
    )
    if not pairs:
        return None
    total = math.fsum(weight for _value, weight in pairs)
    threshold = total * q
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return pairs[-1][0]


def crop_bar_series_to_weighted_quantiles(
    centers: Sequence[float],
    counts: Sequence[float],
    widths: Sequence[float],
    *,
    low_q: float = DEFAULT_BAR_QUANTILE_RANGE[0],
    high_q: float = DEFAULT_BAR_QUANTILE_RANGE[1],
) -> tuple[list[float], list[float], list[float]]:
    """Keep bar bins whose centers fall in the weighted quantile range."""

    if not centers:
        return [], [], []
    low = weighted_quantile(centers, counts, low_q)
    high = weighted_quantile(centers, counts, high_q)
    if low is None or high is None or low > high:
        return list(centers), list(counts), list(widths)
    cropped = [
        (center, count, width)
        for center, count, width in zip(centers, counts, widths, strict=True)
        if low <= center <= high
    ]
    if not cropped:
        return list(centers), list(counts), list(widths)
    return ([item[0] for item in cropped], [item[1] for item in cropped], [item[2] for item in cropped])
