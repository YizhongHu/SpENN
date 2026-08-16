"""Tolerance gates over the He-v1 electron-nucleus cusp and tail metrics.

This module measures nothing. The numeric values it gates are already emitted by
``tpen/evaluation/summaries/atom.py`` -- ``ElectronNucleusCuspSummary`` writes
``cusp_expected_slope``, ``cusp_one_sided_slope_mean`` and the two
``cusp_one_sided_slope_abs_error_*`` keys, and ``ElectronNucleusTailSummary``
writes the ``tail_outer_slope_*``, ``tail_outer_radius_*`` and
``tail_negative_slope_fraction`` keys. Earlier He receipts extracted only the
``*_available`` flags and the ``*_count`` keys, so the values existed and were
never read. This module reads them and compares them against thresholds that the
caller predeclares.

Three invariants are load-bearing and each is covered by a test:

availability is never a value
    An unfitted region yields ``"absent"`` with a reason, never a zero slope. A
    zero parses as data and silently corrupts any median taken over rows.

counts and flags are retained, not replaced
    ``cusp_available``, ``tail_available`` and every ``*_count`` key get their
    own outcome alongside the value gates, so a row that gates green on values
    still carries the evidence of how much data produced them.

the expected cusp slope stays charge-owned
    The Kato condition fixes the one-sided slope at ``-Z`` from the nuclear
    charge in ``spec``. A fitted number never gets to redefine what ``-Z`` means;
    ``cusp_expected_slope`` is itself gated against the declared charge, so a
    summary that drifted away from the charge is caught rather than believed.

Thresholds are parameters. Nothing numeric is decided here: this module is the
mechanism, and the He-v1 production tolerance VALUES are predeclared separately
before the run.

The module is stdlib-only and imports no ``tpen``: ``experiments/README.md``
permits experiment code to import ``tpen`` only as ``tpen.run.run_from_config``
inside launcher scripts, and this is a pure function consumed by the study
collector.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

GateStatus = Literal["pass", "fail", "absent"]


@dataclass(frozen=True)
class GateOutcome:
    """One gate's verdict on one evaluation metric.

    Parameters
    ----------
    name : str
        Stable identifier of the gate, unique within one evaluation. Named for
        the metric it reads plus the comparison it applies, so two gates over
        opposite bounds of the same metric stay distinguishable in a receipt.
    status : {"pass", "fail", "absent"}
        ``"absent"`` means the gate could not be decided -- the fit was
        unavailable, the metric was not emitted, or its threshold was not
        declared. It never means "passed with no data".
    value : float or bool or None
        The observed metric value, ``None`` when the metric was unavailable.
        Boolean for the availability flags, float otherwise.
    threshold : float or bool or None
        The predeclared bound this value was compared against, ``None`` when no
        threshold was declared.
    reason : str
        Human-readable justification, always populated -- including on
        ``"pass"``, so a receipt row explains itself without re-deriving the
        comparison.
    """

    name: str
    status: GateStatus
    value: float | bool | None
    threshold: float | bool | None
    reason: str


@dataclass(frozen=True)
class _GateDefinition:
    """Declarative description of a single gate.

    Attributes
    ----------
    name : str
        Outcome name.
    metric : str
        Key read from the flat evaluation metrics mapping.
    comparison : {"at_most", "at_least", "near_expected_cusp_slope", "required_flag"}
        How ``metric`` is compared against its threshold.
    threshold_key : str
        Key read from ``spec``.
    availability_key : str or None
        Availability flag that must be true before the metric can carry a value.
        ``None`` for metrics that ``atom.py`` emits unconditionally (the flags
        and the counts), which is what keeps counts readable on an unfitted row.
    rationale : str
        Why this bound exists, quoted into the outcome reason.
    """

    name: str
    metric: str
    comparison: str
    threshold_key: str
    availability_key: str | None
    rationale: str


#: Charge the expected cusp slope is derived from. Owned by the physical system,
#: never by a fit.
_CHARGE_KEY = "nuclear_charge"

_CUSP_AVAILABLE = "cusp_available"
_TAIL_AVAILABLE = "tail_available"

_GATE_DEFINITIONS: tuple[_GateDefinition, ...] = (
    # --- cusp: availability and counts, always readable -------------------
    _GateDefinition(
        name="cusp_available",
        metric=_CUSP_AVAILABLE,
        comparison="required_flag",
        threshold_key="require_cusp_available",
        availability_key=None,
        rationale="a production row with no cusp fit has no cusp evidence",
    ),
    _GateDefinition(
        name="cusp_finite_fit_count_at_least",
        metric="cusp_finite_fit_count",
        comparison="at_least",
        threshold_key="cusp_finite_fit_count_min",
        availability_key=None,
        rationale="too few finite fits makes the cusp means unrepresentative",
    ),
    _GateDefinition(
        name="cusp_finite_measurement_count_at_least",
        metric="cusp_finite_measurement_count",
        comparison="at_least",
        threshold_key="cusp_finite_measurement_count_min",
        availability_key=None,
        rationale="finite measurements are the raw support behind every cusp fit",
    ),
    # --- cusp: values, charge-referenced -----------------------------------
    _GateDefinition(
        name="cusp_expected_slope_near_charge",
        metric="cusp_expected_slope",
        comparison="near_expected_cusp_slope",
        threshold_key="cusp_expected_slope_tolerance",
        availability_key=_CUSP_AVAILABLE,
        rationale="the summary's expected slope must still be the charge-owned -Z",
    ),
    _GateDefinition(
        name="cusp_one_sided_slope_mean_near_charge",
        metric="cusp_one_sided_slope_mean",
        comparison="near_expected_cusp_slope",
        threshold_key="cusp_one_sided_slope_mean_abs_error_max",
        availability_key=_CUSP_AVAILABLE,
        rationale="the mean fitted slope must sit near the Kato value -Z",
    ),
    _GateDefinition(
        name="cusp_one_sided_slope_abs_error_mean_at_most",
        metric="cusp_one_sided_slope_abs_error_mean",
        comparison="at_most",
        threshold_key="cusp_one_sided_slope_abs_error_mean_max",
        availability_key=_CUSP_AVAILABLE,
        rationale="average per-fit cusp error, which cancellation cannot hide",
    ),
    _GateDefinition(
        name="cusp_one_sided_slope_abs_error_max_at_most",
        metric="cusp_one_sided_slope_abs_error_max",
        comparison="at_most",
        threshold_key="cusp_one_sided_slope_abs_error_max_max",
        availability_key=_CUSP_AVAILABLE,
        rationale="worst single fit, so one broken ray cannot average away",
    ),
    # --- tail: availability and counts, always readable --------------------
    _GateDefinition(
        name="tail_available",
        metric=_TAIL_AVAILABLE,
        comparison="required_flag",
        threshold_key="require_tail_available",
        availability_key=None,
        rationale="a production row with no tail fit has no decay evidence",
    ),
    _GateDefinition(
        name="tail_outer_measurement_count_at_least",
        metric="tail_outer_measurement_count",
        comparison="at_least",
        threshold_key="tail_outer_measurement_count_min",
        availability_key=None,
        rationale="one outer ray per group; too few makes the sign fraction coarse",
    ),
    _GateDefinition(
        name="tail_finite_measurement_count_at_least",
        metric="tail_finite_measurement_count",
        comparison="at_least",
        threshold_key="tail_finite_measurement_count_min",
        availability_key=None,
        rationale="finite measurements are the raw support behind every outer ray",
    ),
    # --- tail: sign -------------------------------------------------------
    _GateDefinition(
        name="tail_negative_slope_fraction_at_least",
        metric="tail_negative_slope_fraction",
        comparison="at_least",
        threshold_key="tail_negative_slope_fraction_min",
        availability_key=_TAIL_AVAILABLE,
        rationale="a bound state decays, so outer rays must slope negative",
    ),
    _GateDefinition(
        name="tail_outer_slope_max_at_most",
        metric="tail_outer_slope_max",
        comparison="at_most",
        threshold_key="tail_outer_slope_max_max",
        availability_key=_TAIL_AVAILABLE,
        rationale="the least-negative ray must still be decaying, not growing",
    ),
    # --- tail: magnitude ---------------------------------------------------
    _GateDefinition(
        name="tail_outer_slope_mean_at_most",
        metric="tail_outer_slope_mean",
        comparison="at_most",
        threshold_key="tail_outer_slope_mean_max",
        availability_key=_TAIL_AVAILABLE,
        rationale="mean decay must be at least as fast as the declared floor",
    ),
    _GateDefinition(
        name="tail_outer_slope_mean_at_least",
        metric="tail_outer_slope_mean",
        comparison="at_least",
        threshold_key="tail_outer_slope_mean_min",
        availability_key=_TAIL_AVAILABLE,
        rationale="an implausibly steep mean slope signals underflow, not physics",
    ),
    _GateDefinition(
        name="tail_outer_slope_min_at_least",
        metric="tail_outer_slope_min",
        comparison="at_least",
        threshold_key="tail_outer_slope_min_min",
        availability_key=_TAIL_AVAILABLE,
        rationale="bounds the steepest single ray, where amplitudes underflow first",
    ),
    # --- tail: where the slope was measured --------------------------------
    _GateDefinition(
        name="tail_outer_radius_min_at_least",
        metric="tail_outer_radius_min",
        comparison="at_least",
        threshold_key="tail_outer_radius_min_min",
        availability_key=_TAIL_AVAILABLE,
        rationale="the nearest outer ray must still reach the asymptotic region",
    ),
    _GateDefinition(
        name="tail_outer_radius_max_at_most",
        metric="tail_outer_radius_max",
        comparison="at_most",
        threshold_key="tail_outer_radius_max_max",
        availability_key=_TAIL_AVAILABLE,
        rationale="past the declared radius the amplitude is numerical noise",
    ),
)

#: Metric keys this module reads. A collector should retain all of them in the
#: receipt, values and evidence alike, not only the ones that happened to gate.
ATOM_GATE_METRIC_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(definition.metric for definition in _GATE_DEFINITIONS)
)

#: Threshold keys ``spec`` may declare. Anything else is a typo, not a setting.
ATOM_GATE_SPEC_KEYS: tuple[str, ...] = (_CHARGE_KEY,) + tuple(
    dict.fromkeys(definition.threshold_key for definition in _GATE_DEFINITIONS)
)


def evaluate_atom_gates(
    metrics: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
) -> tuple[GateOutcome, ...]:
    """Gate the emitted cusp and tail metrics against predeclared tolerances.

    Parameters
    ----------
    metrics : Mapping[str, Any]
        Flat evaluation metrics mapping, as emitted by the atom summaries. Keys
        absent from the mapping are reported ``"absent"``, never defaulted.
    spec : Mapping[str, Any]
        Tolerance mapping. Recognized keys are ``ATOM_GATE_SPEC_KEYS``; an
        undeclared threshold yields an ``"absent"`` outcome rather than a silent
        skip, so an unset tolerance can never read as a pass.

    Returns
    -------
    tuple of GateOutcome
        One outcome per gate, in a fixed order: cusp availability, cusp counts,
        cusp values, then the same for the tail.

    Raises
    ------
    ValueError
        If ``spec`` carries a key outside ``ATOM_GATE_SPEC_KEYS``. A mistyped
        threshold key would otherwise disable exactly the gate it was meant to
        set, and disable it quietly.

    Notes
    -----
    Failure is closed. A metric present but non-finite, wrongly typed, or
    outside its bound is ``"fail"``; only genuinely missing data is ``"absent"``.
    """

    unknown = sorted(set(spec) - set(ATOM_GATE_SPEC_KEYS))
    if unknown:
        raise ValueError(
            "unknown atom gate spec keys "
            f"{unknown}; recognized keys are {list(ATOM_GATE_SPEC_KEYS)}"
        )
    return tuple(_evaluate(definition, metrics, spec) for definition in _GATE_DEFINITIONS)


def _evaluate(
    definition: _GateDefinition,
    metrics: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> GateOutcome:
    """Decide one gate."""

    # Availability first: an unfitted region has no value to compare, and
    # inventing one (zero, or the threshold itself) is the failure this ordering
    # exists to prevent.
    if definition.availability_key is not None:
        available = metrics.get(definition.availability_key)
        if available is None:
            return _absent(
                definition,
                f"availability flag '{definition.availability_key}' is absent from the metrics",
            )
        if not isinstance(available, bool):
            return _fail(
                definition,
                None,
                None,
                f"availability flag '{definition.availability_key}' is "
                f"{available!r}, which is not a boolean",
            )
        if not available:
            return _absent(
                definition,
                f"fit unavailable: '{definition.availability_key}' is False, so "
                f"'{definition.metric}' carries no value",
            )

    if definition.metric not in metrics:
        return _absent(definition, f"metric '{definition.metric}' is absent from the metrics")
    raw = metrics[definition.metric]

    if definition.comparison == "required_flag":
        return _evaluate_flag(definition, raw, spec)

    threshold = _threshold(definition, spec)
    if threshold is None:
        # Undecidable, but the measurement is real: retain it so an ungated run
        # still reports what was observed.
        return GateOutcome(
            name=definition.name,
            status="absent",
            value=_numeric(raw),
            threshold=None,
            reason=(
                f"threshold '{definition.threshold_key}' is not declared in the spec, "
                f"so '{definition.metric}' = {raw!r} is retained ungated"
            ),
        )
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError(
            f"spec['{definition.threshold_key}'] must be a real number, got {threshold!r}"
        )
    threshold = float(threshold)

    value = _numeric(raw)
    if value is None:
        return _fail(
            definition,
            None,
            threshold,
            f"metric '{definition.metric}' is {raw!r}, which is not a real number",
        )
    if not isfinite(value):
        return _fail(
            definition,
            value,
            threshold,
            f"metric '{definition.metric}' is {value!r}, which is not finite",
        )

    if definition.comparison == "near_expected_cusp_slope":
        return _evaluate_near_charge(definition, value, threshold, spec)
    return _evaluate_bound(definition, value, threshold)


def _evaluate_flag(
    definition: _GateDefinition,
    raw: Any,
    spec: Mapping[str, Any],
) -> GateOutcome:
    """Decide an availability-requirement gate.

    The flag is retained as an outcome in its own right so a receipt records
    that the fit existed, separately from what the fit said.
    """

    if not isinstance(raw, bool):
        return _fail(
            definition,
            None,
            None,
            f"metric '{definition.metric}' is {raw!r}, which is not a boolean",
        )
    required = _threshold(definition, spec)
    if required is None:
        return GateOutcome(
            name=definition.name,
            status="absent",
            value=raw,
            threshold=None,
            reason=(
                f"requirement '{definition.threshold_key}' is not declared in the spec; "
                f"'{definition.metric}' is {raw} and is retained ungated"
            ),
        )
    if not isinstance(required, bool):
        raise ValueError(
            f"spec['{definition.threshold_key}'] must be a boolean, got {required!r}"
        )
    if raw:
        return GateOutcome(
            name=definition.name,
            status="pass",
            value=True,
            threshold=required,
            reason=f"'{definition.metric}' is True ({definition.rationale})",
        )
    if required:
        return GateOutcome(
            name=definition.name,
            status="fail",
            value=False,
            threshold=required,
            reason=(
                f"'{definition.metric}' is False but the spec requires it "
                f"({definition.rationale})"
            ),
        )
    return GateOutcome(
        name=definition.name,
        status="absent",
        value=False,
        threshold=required,
        reason=(
            f"'{definition.metric}' is False and the spec does not require it; "
            "no value gates for this region can be decided"
        ),
    )


def _evaluate_near_charge(
    definition: _GateDefinition,
    value: float,
    tolerance: float,
    spec: Mapping[str, Any],
) -> GateOutcome:
    """Compare a slope against the charge-owned expected cusp slope ``-Z``.

    The target comes from ``spec['nuclear_charge']`` and from nowhere else. A
    fitted slope is the thing under test; it is never allowed to supply its own
    reference.
    """

    charge = spec.get(_CHARGE_KEY)
    if charge is None:
        return GateOutcome(
            name=definition.name,
            status="absent",
            value=value,
            threshold=tolerance,
            reason=(
                f"'{_CHARGE_KEY}' is not declared in the spec, so the charge-owned "
                f"expected slope -Z is unknown and '{definition.metric}' = {value!r} "
                "is retained ungated"
            ),
        )
    if not isinstance(charge, (int, float)) or isinstance(charge, bool):
        raise ValueError(f"spec['{_CHARGE_KEY}'] must be a real number, got {charge!r}")
    charge = float(charge)
    if not isfinite(charge) or charge <= 0.0:
        raise ValueError(f"spec['{_CHARGE_KEY}'] must be positive and finite, got {charge!r}")

    expected = -charge
    error = abs(value - expected)
    status: GateStatus = "pass" if error <= tolerance else "fail"
    comparator = "within" if status == "pass" else "outside"
    return GateOutcome(
        name=definition.name,
        status=status,
        value=value,
        threshold=tolerance,
        reason=(
            f"'{definition.metric}' = {value!r} is {comparator} {tolerance!r} of the "
            f"charge-owned expected slope -Z = {expected!r} "
            f"(Z = {charge!r}, |error| = {error!r}; {definition.rationale})"
        ),
    )


def _evaluate_bound(
    definition: _GateDefinition,
    value: float,
    threshold: float,
) -> GateOutcome:
    """Compare a value against a one-sided bound."""

    if definition.comparison == "at_most":
        satisfied = value <= threshold
        symbol = "<="
    elif definition.comparison == "at_least":
        satisfied = value >= threshold
        symbol = ">="
    else:  # pragma: no cover - guarded by the definition table above
        raise ValueError(f"unknown comparison {definition.comparison!r}")
    status: GateStatus = "pass" if satisfied else "fail"
    negation = "" if satisfied else "not "
    return GateOutcome(
        name=definition.name,
        status=status,
        value=value,
        threshold=threshold,
        reason=(
            f"'{definition.metric}' = {value!r} is {negation}{symbol} {threshold!r} "
            f"({definition.rationale})"
        ),
    )


def _threshold(definition: _GateDefinition, spec: Mapping[str, Any]) -> Any:
    """Return the declared threshold, or ``None`` when the spec omits it."""

    return spec.get(definition.threshold_key)


def _numeric(raw: Any) -> float | None:
    """Coerce a metric to ``float``, rejecting bools and non-numbers.

    ``bool`` is a subclass of ``int``, so an availability flag landing in a
    numeric slot would otherwise silently compare as 0 or 1.
    """

    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def _absent(definition: _GateDefinition, reason: str) -> GateOutcome:
    """Build an undecidable outcome. Never carries a substituted value."""

    return GateOutcome(
        name=definition.name,
        status="absent",
        value=None,
        threshold=None,
        reason=reason,
    )


def _fail(
    definition: _GateDefinition,
    value: float | bool | None,
    threshold: float | bool | None,
    reason: str,
) -> GateOutcome:
    """Build a failing outcome for malformed or out-of-bound data."""

    return GateOutcome(
        name=definition.name,
        status="fail",
        value=value,
        threshold=threshold,
        reason=reason,
    )


__all__ = [
    "ATOM_GATE_METRIC_KEYS",
    "ATOM_GATE_SPEC_KEYS",
    "GateOutcome",
    "GateStatus",
    "evaluate_atom_gates",
]
