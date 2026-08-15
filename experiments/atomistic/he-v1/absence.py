"""Explicit absence: a missing value is never zero and never blank.

A blank CSV cell parses to NaN and is then silently dropped, so a median over
two of nine rows renders exactly like a median over all nine. A zero is worse:
it parses as data and drags the aggregate toward it. This module owns the one
sentinel every stage of the study uses instead, and the rendering that keeps it
visible in JSON, CSV, and the report.

Aggregates built through :func:`summarize_values` always carry how many rows
actually supplied a value, so "median over 9 rows" can never be read off a
statistic computed from 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

#: Text used for an absent value everywhere a human or a parser can see it.
ABSENT_TEXT = "absent"


class _Absent:
    """Singleton marker for a value that was not measured.

    Deliberately not falsy-friendly beyond ``bool(...) is False``: any code that
    treats it as a number raises rather than silently contributing a zero.
    """

    _instance: "_Absent | None" = None

    def __new__(cls) -> "_Absent":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return ABSENT_TEXT

    def __str__(self) -> str:
        return ABSENT_TEXT

    def __bool__(self) -> bool:
        return False

    def __float__(self) -> float:
        raise TypeError("absent values have no numeric value; render them as 'absent'")


#: The single absence marker.
ABSENT = _Absent()


def is_absent(value: Any) -> bool:
    """Return whether ``value`` is the absence marker."""

    return value is ABSENT


def present_or_absent(value: Any) -> Any:
    """Return ``value``, mapping ``None`` and non-finite floats to :data:`ABSENT`.

    ``None`` is what a missing key looks like; NaN and infinity are what a
    broken measurement looks like. Neither is a number a receipt may aggregate,
    and both would otherwise survive into a mean.
    """

    if value is None or is_absent(value):
        return ABSENT
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return ABSENT
    return value


def cell(value: Any) -> dict[str, Any]:
    """Return one JSON cell that states its own presence.

    A consumer never has to guess whether ``null`` meant "not measured" or
    "measured as null": ``status`` says so.
    """

    resolved = present_or_absent(value)
    if is_absent(resolved):
        return {"status": ABSENT_TEXT, "value": None}
    return {"status": "present", "value": resolved}


def render(value: Any) -> str:
    """Render one value for CSV or a text report.

    An absent value renders as the literal ``absent``. It is never rendered as
    an empty cell, and never as ``0``.
    """

    resolved = present_or_absent(value)
    if is_absent(resolved):
        return ABSENT_TEXT
    if isinstance(resolved, bool):
        return "true" if resolved else "false"
    return str(resolved)


def cell_value(payload: Any) -> Any:
    """Return the value carried by a serialized :func:`cell`, or :data:`ABSENT`."""

    if isinstance(payload, Mapping):
        if str(payload.get("status")) == ABSENT_TEXT:
            return ABSENT
        return present_or_absent(payload.get("value"))
    return present_or_absent(payload)


@dataclass(frozen=True)
class ValueSummary:
    """An aggregate that carries its own coverage.

    Parameters
    ----------
    n_rows : int
        Rows the aggregate was asked about.
    n_present : int
        Rows that actually supplied a value.
    n_absent : int
        Rows that did not.
    mean, median_value, minimum, maximum : float or _Absent
        Statistics over the present values only, or :data:`ABSENT` when no row
        supplied one. An aggregate over zero rows is absent, not zero.
    """

    n_rows: int
    n_present: int
    n_absent: int
    mean: Any
    median_value: Any
    minimum: Any
    maximum: Any

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping with explicit coverage."""

        return {
            "n_rows": self.n_rows,
            "n_present": self.n_present,
            "n_absent": self.n_absent,
            "mean": cell(self.mean),
            "median": cell(self.median_value),
            "min": cell(self.minimum),
            "max": cell(self.maximum),
        }

    def coverage_text(self) -> str:
        """Return the human-readable coverage, e.g. ``2/9 rows``."""

        return f"{self.n_present}/{self.n_rows} rows"


def summarize_values(values: Sequence[Any] | Iterable[Any]) -> ValueSummary:
    """Aggregate ``values``, keeping absent rows visible.

    Absent entries are excluded from the statistics and counted, so a caller
    can always tell how much of the grid an aggregate actually rests on.
    """

    rows = [present_or_absent(value) for value in values]
    present = [float(value) for value in rows if not is_absent(value) and not isinstance(value, str)]
    n_rows = len(rows)
    n_present = len(present)
    if not present:
        return ValueSummary(
            n_rows=n_rows,
            n_present=0,
            n_absent=n_rows,
            mean=ABSENT,
            median_value=ABSENT,
            minimum=ABSENT,
            maximum=ABSENT,
        )
    return ValueSummary(
        n_rows=n_rows,
        n_present=n_present,
        n_absent=n_rows - n_present,
        mean=sum(present) / n_present,
        median_value=median(present),
        minimum=min(present),
        maximum=max(present),
    )


__all__ = [
    "ABSENT",
    "ABSENT_TEXT",
    "ValueSummary",
    "cell",
    "cell_value",
    "is_absent",
    "present_or_absent",
    "render",
    "summarize_values",
]
