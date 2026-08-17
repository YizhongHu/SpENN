"""Discharge the predeclared convergence-assessment rule on one training run.

`production_grid.yaml` declares the rule; nothing in the pipeline executes it.
This script is the artifact that makes the declaration dischargeable: H-F3 runs
it on the stage-1 loss trace between stages, and its verdict is recorded in the
H-F3 receipt before stage-2 rows are submitted.

IT REPORTS AND NEVER RE-SELECTS. No branch here chooses an arm, a checkpoint, a
seed or a budget. `may_reselect: false` is enforced by `plan.py`; this script is
written so that constraint is visible in its structure rather than merely
promised -- the only outputs are a verdict string and the numbers behind it.

THE RULE PARAMETERS ARE READ FROM THE GRID, NOT FROM THE COMMAND LINE. A
predeclared rule that could be re-parameterised at run time by the person
reading the result is not predeclared. There is deliberately no flag to change
`n_windows`, `n_trailing_windows`, or `window_width_min_tau_multiple`.

WHY A SIGN TEST RATHER THAN A TAIL AVERAGE OR AN ERROR-BAR OVERLAP TEST: both
alternatives were measured in a peer lane and both failed toward false
reassurance. A tail average hid a run that spent its entire budget inside its
optimization transient. An overlap test passes a series drifting monotonically
inside its own bars -- measured at five windows drifting 37 uHa against 10 uHa
bars, where every adjacent pair overlaps while the cumulative drift is 3.70
bars. The mirror case scattered 5.8 bars and was pure noise, because it was
non-monotone. Bar magnitude is unreliable in BOTH directions; the sign pattern
is the discriminator, which is why cumulative drift is reported here as context
and decides nothing.

RUN THIS IN A SLURM ALLOCATION, NOT ON A LOGIN NODE. It is cheap enough to
tempt otherwise. It gates roughly 97 GPU-hours, and H-F3's receipt is required
to carry its full output, so it deserves a job ID and a durable log on its own
merits. That also makes the torch import below a non-issue rather than a
tolerated exception.

THIS MODULE IMPORTS ``tpen``, unlike `plan.py`, `launch.py` and `collect.py`,
and does so LAZILY inside :func:`loss_series_tau` so the module still imports on
a login node. It is the second sanctioned exception to the ``experiments/``
import boundary after `driver.py`, and `test_experiments_import_boundary_holds`
enumerates it explicitly rather than relaxing the rule.

The exception is necessary rather than convenient: ``tpen.statistics`` is the
single sanctioned producer of ``tau_int`` in this repository, and
re-implementing Geyer's initial positive sequence here would be a second
estimator that could silently drift from the one the study reports. The rule
itself -- 20 windows, 8 trailing, a 5x tau multiple, three verdicts -- is He-v1
PREDECLARATION POLICY and not a general TPEN capability, which is why the policy
lives here with the study while the estimator stays in the library.

TWO-SIDED DETECTION, THREE-WAY CONSEQUENCE. The declared false-alarm rate is
``2 * (1/2)**(n-1)`` = 1.56% at eight windows, and the leading factor of two
makes it two-sided: the test fires on any all-same-sign run of seven
differences. But the two signs mean different things and are NOT merged:

- monotone DESCENT -> ``budget_inadequate``. The model was still improving when
  the budget ran out. Report as a stated limitation on the result's
  interpretation, keep the final checkpoint as primary, take no extension.
- monotone ASCENT -> ``training_pathology``. The loss was RISING at the end of
  training, which extending the budget would make worse, not better. That is
  divergence or an unstable step size, and it invalidates the run rather than
  annotating it. ESCALATE.
- neither -> ``adequate``.

The 1.56% splits evenly at 0.78% per direction.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as _stats
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VERDICT_ADEQUATE = "adequate"
VERDICT_BUDGET_INADEQUATE = "budget_inadequate"
VERDICT_TRAINING_PATHOLOGY = "training_pathology"
VERDICT_INDETERMINATE = "indeterminate"

DEFAULT_LOSS_KEY = "loss"


class AssessmentError(RuntimeError):
    """The assessment cannot be performed as declared."""


@dataclass(frozen=True)
class ConvergenceAssessment:
    """One assessment, carrying every number behind its verdict.

    Attributes
    ----------
    verdict : str
        ``adequate``, ``budget_inadequate``, ``training_pathology`` or
        ``indeterminate``.
    direction : str or None
        ``descending`` or ``ascending`` when the sign test fired, else ``None``.
    reason : str
        Why this verdict, in words.
    n_samples : int
        Loss values read.
    tau : float or None
        Measured autocorrelation time of the LOSS SERIES over the trailing
        region. This is not the local-energy tau: the loss is a trajectory
        through parameter space under an optimizer and its correlation time is a
        different, generally longer quantity.
    window_width : int
        Samples per window, fixed by ``n_windows`` and the series length.
    tau_multiple : float or None
        ``window_width / tau``, the quantity compared against the declared
        minimum.
    required_tau_multiple : float
        The declared minimum from the grid.
    window_means : list of float
        All ``n_windows`` window means.
    trailing_means : list of float
        The final ``n_trailing_windows`` means the sign test reads.
    signs : list of int
        Signs of the consecutive differences of ``trailing_means``.
    cumulative_drift_bars : float or None
        Drift across the trailing region in units of one window mean's standard
        error. CONTEXT ONLY -- it decides nothing.
    """

    verdict: str
    direction: str | None
    reason: str
    n_samples: int
    tau: float | None
    window_width: int
    tau_multiple: float | None
    required_tau_multiple: float
    window_means: list[float] = field(default_factory=list)
    trailing_means: list[float] = field(default_factory=list)
    signs: list[int] = field(default_factory=list)
    cumulative_drift_bars: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping for the H-F3 receipt."""

        return {
            "verdict": self.verdict,
            "direction": self.direction,
            "reason": self.reason,
            "n_samples": self.n_samples,
            "loss_series_tau": self.tau,
            "window_width": self.window_width,
            "tau_multiple": self.tau_multiple,
            "required_tau_multiple": self.required_tau_multiple,
            "window_means": self.window_means,
            "trailing_means": self.trailing_means,
            "signs": self.signs,
            "cumulative_drift_bars": self.cumulative_drift_bars,
        }


def read_rule(grid_path: str | Path) -> dict[str, Any]:
    """Read the predeclared `convergence_assessment` block from the grid."""

    with Path(grid_path).open(encoding="utf-8") as stream:
        grid = yaml.safe_load(stream)
    if not isinstance(grid, dict) or "convergence_assessment" not in grid:
        raise AssessmentError(f"{grid_path} carries no convergence_assessment block")
    rule = grid["convergence_assessment"]
    if rule.get("method") != "windowed_means_sign_test":
        raise AssessmentError(
            f"this script implements windowed_means_sign_test; the grid declares "
            f"{rule.get('method')!r}"
        )
    if rule.get("may_reselect") is not False:
        raise AssessmentError(
            "convergence_assessment.may_reselect must be false; this script only reports"
        )
    return rule


def read_loss_series(
    metrics_path: str | Path, *, key: str = DEFAULT_LOSS_KEY
) -> list[float]:
    """Read one metric series from a run's ``metrics.jsonl``, in step order.

    The JSONL logger writes ``{"step": int, "namespace": str, "metrics": {...}}``
    per line. Records are sorted by step rather than trusted to be in order,
    because an out-of-order series would silently fabricate sign changes.
    """

    path = Path(metrics_path)
    rows: list[tuple[int, float]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metrics = record.get("metrics") or {}
            if key in metrics:
                rows.append((int(record["step"]), float(metrics[key])))
    if not rows:
        raise AssessmentError(f"no metric {key!r} found in {path}")
    rows.sort(key=lambda item: item[0])
    return [value for _, value in rows]


def loss_series_tau(values: Sequence[float]) -> tuple[float | None, str | None]:
    """Return the sanctioned tau estimate for a scalar series, or its refusal.

    Delegates to :func:`tpen.statistics.autocorrelation.integrated_autocorrelation_time`
    with the series presented as a single ``[draw, 1]`` chain. TPEN uses Geyer's
    initial positive sequence; there is no blocking estimator here and no second
    implementation of this one.

    The imports are function-local so this module still loads on a login node
    where torch is unavailable, matching `driver.py`'s sanctioned pattern.
    """

    import torch  # noqa: PLC0415 - sanctioned boundary exception, see module docstring

    from tpen.statistics.autocorrelation import (  # noqa: PLC0415 - sanctioned exception
        integrated_autocorrelation_time,
    )

    tensor = torch.tensor(list(values), dtype=torch.float64).unsqueeze(1)
    result = integrated_autocorrelation_time(tensor)
    return result.tau_int, result.reason


def assess(
    values: Sequence[float],
    *,
    n_windows: int,
    n_trailing_windows: int,
    window_width_min_tau_multiple: float,
) -> ConvergenceAssessment:
    """Apply the predeclared windowed-means sign test to one loss series."""

    n_samples = len(values)
    window_width = n_samples // n_windows

    def refuse(reason: str, *, tau: float | None = None) -> ConvergenceAssessment:
        return ConvergenceAssessment(
            verdict=VERDICT_INDETERMINATE,
            direction=None,
            reason=reason,
            n_samples=n_samples,
            tau=tau,
            window_width=window_width,
            tau_multiple=None if tau in (None, 0.0) else window_width / tau,
            required_tau_multiple=window_width_min_tau_multiple,
        )

    if window_width < 1:
        return refuse(
            f"{n_samples} samples cannot fill {n_windows} windows; the rule declares "
            f"{n_windows} windows and this script does not reduce them to fit"
        )

    # Windows are fixed by the DECLARED n_windows and the series length. tau
    # validates that width; it never chooses it, so there is no circularity.
    windows = [
        values[index * window_width : (index + 1) * window_width] for index in range(n_windows)
    ]
    window_means = [float(sum(window) / len(window)) for window in windows]
    trailing_means = window_means[-n_trailing_windows:]

    # tau is measured on the TRAILING REGION, which is the span the sign test
    # actually reads. Measuring it over the whole trace would fold the
    # optimization transient into the correlation estimate and report a tau that
    # describes a regime the test does not examine.
    trailing_values = values[(n_windows - n_trailing_windows) * window_width :]
    tau, tau_reason = loss_series_tau(trailing_values)
    if tau is None:
        return refuse(
            "the loss-series autocorrelation time did not resolve on the trailing region, "
            f"so window independence cannot be established: {tau_reason}"
        )

    tau_multiple = window_width / tau
    if tau_multiple < window_width_min_tau_multiple:
        # REFUSE rather than shrink. Narrowing the windows to satisfy the
        # multiple would raise the count of same-sign runs under the null and
        # make the declared 1.56% false-alarm rate optimistic by an unknown
        # factor -- a gate whose error rate nobody has bounded.
        supportable = int(n_samples // (window_width_min_tau_multiple * tau))
        return refuse(
            f"window width {window_width} is {tau_multiple:.2f}x the measured loss-series "
            f"tau {tau:.2f}, below the declared minimum "
            f"{window_width_min_tau_multiple}x. This series supports at most "
            f"{supportable} windows at that multiple, fewer than the declared "
            f"{n_windows}. The windows are NOT narrowed to fit: escalate instead",
            tau=tau,
        )

    differences = [
        trailing_means[index + 1] - trailing_means[index]
        for index in range(len(trailing_means) - 1)
    ]
    signs = [0 if difference == 0.0 else (1 if difference > 0.0 else -1) for difference in differences]

    # One window mean's standard error, inflated for correlation. Reported for
    # context and used by no branch below.
    variance = _stats.variance(float(value) for value in trailing_values)
    bar = math.sqrt(variance * tau / window_width) if window_width else float("nan")
    drift = trailing_means[-1] - trailing_means[0]
    drift_bars = drift / bar if bar > 0.0 else None

    common = {
        "n_samples": n_samples,
        "tau": tau,
        "window_width": window_width,
        "tau_multiple": tau_multiple,
        "required_tau_multiple": window_width_min_tau_multiple,
        "window_means": window_means,
        "trailing_means": trailing_means,
        "signs": signs,
        "cumulative_drift_bars": drift_bars,
    }

    if signs and all(sign == -1 for sign in signs):
        return ConvergenceAssessment(
            verdict=VERDICT_BUDGET_INADEQUATE,
            direction="descending",
            reason=(
                f"all {len(signs)} consecutive differences of the trailing "
                f"{len(trailing_means)} window means are negative: the loss was still "
                "descending when the budget ran out. Report as a stated limitation on "
                "the result's interpretation; take no extension"
            ),
            **common,
        )
    if signs and all(sign == 1 for sign in signs):
        return ConvergenceAssessment(
            verdict=VERDICT_TRAINING_PATHOLOGY,
            direction="ascending",
            reason=(
                f"all {len(signs)} consecutive differences of the trailing "
                f"{len(trailing_means)} window means are positive: the loss was RISING at "
                "the end of training. This is not a budget shortfall -- extending it would "
                "make matters worse -- but divergence or an unstable step size. ESCALATE"
            ),
            **common,
        )
    return ConvergenceAssessment(
        verdict=VERDICT_ADEQUATE,
        direction=None,
        reason=(
            f"the {len(signs)} consecutive differences of the trailing "
            f"{len(trailing_means)} window means do not share a sign, so no monotone "
            "trend is detected at the declared false-alarm rate"
        ),
        **common,
    )


def render(assessment: ConvergenceAssessment) -> str:
    """Render one assessment as a human-readable report."""

    lines = [
        "He-v1 convergence assessment (windowed means, sign test)",
        f"  verdict                  : {assessment.verdict.upper()}",
        f"  direction                : {assessment.direction or 'none'}",
        f"  reason                   : {assessment.reason}",
        f"  samples                  : {assessment.n_samples}",
        f"  loss-series tau          : {assessment.tau}",
        f"  window width             : {assessment.window_width}",
        f"  width / tau              : {assessment.tau_multiple}",
        f"  required width / tau     : {assessment.required_tau_multiple}",
    ]
    if assessment.window_means:
        lines.append(f"  trailing window means    : {assessment.trailing_means}")
        lines.append(f"  consecutive-difference signs : {assessment.signs}")
        lines.append(
            f"  cumulative drift (bars)  : {assessment.cumulative_drift_bars}  "
            "[CONTEXT ONLY -- decides nothing]"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Assess one run and print the report; exit nonzero unless adequate."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("metrics", help="path to the training run's metrics.jsonl")
    parser.add_argument(
        "--grid",
        default=str(Path(__file__).resolve().parent / "configs" / "production_grid.yaml"),
        help="grid config carrying the predeclared convergence_assessment block",
    )
    parser.add_argument("--key", default=DEFAULT_LOSS_KEY, help="metric key of the loss series")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = parser.parse_args(argv)

    rule = read_rule(args.grid)
    values = read_loss_series(args.metrics, key=args.key)
    assessment = assess(
        values,
        n_windows=int(rule["n_windows"]),
        n_trailing_windows=int(rule["n_trailing_windows"]),
        window_width_min_tau_multiple=float(rule["window_width_min_tau_multiple"]),
    )

    print(json.dumps(assessment.to_dict(), indent=2) if args.json else render(assessment))

    # A nonzero status marks "needs a human", never an action taken here.
    return 0 if assessment.verdict == VERDICT_ADEQUATE else 1


if __name__ == "__main__":
    sys.exit(main())
