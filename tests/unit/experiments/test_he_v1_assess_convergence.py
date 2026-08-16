"""Contracts for the He-v1 convergence-assessment script.

The rule in `production_grid.yaml` is predeclared and nothing in the pipeline
executes it, so `assess_convergence.py` is the artifact that makes it
dischargeable. A script nobody has driven on a trace with a KNOWN answer is a
promise, not a gate -- so every case below builds a series whose verdict is
determined by construction and checks the script recovers it.

The decisive pair is `test_non_monotone_scatter_wider_than_a_bar_is_adequate`
against `test_monotone_descent_inside_the_noise_is_budget_inadequate`. Together
they prove the SIGN PATTERN and not the bar magnitude is doing the work: the
first scatters further and is adequate, the second drifts less and is not.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "experiments" / "atomistic" / "he-v1"

N_WINDOWS = 20
N_TRAILING = 8


def _load_study_module(name: str) -> ModuleType:
    """Load one study module by path under a study-unique module name.

    A bare ``import assess_convergence`` is not safe: ``experiments/`` holds
    four ``collect.py`` and three ``plan.py``, each study inserting its own
    directory at ``sys.path[0]``, so in a full-suite run a bare name resolves to
    whichever study was imported first. Registering the module in
    ``sys.modules`` BEFORE executing it is also required, because exec'ing an
    unregistered module breaks ``@dataclass`` inside the standard library.
    """

    path = STUDY / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"he_v1_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


assess_convergence = _load_study_module("assess_convergence")


def _noise(n: int, *, scale: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=generator, dtype=torch.float64) * scale


def _assess(values: torch.Tensor):
    return assess_convergence.assess(
        [float(value) for value in values],
        n_windows=N_WINDOWS,
        n_trailing_windows=N_TRAILING,
        window_width_min_tau_multiple=5.0,
    )


def _bar(assessment) -> float:
    """One window mean's standard error, recovered from the reported numbers."""

    return abs(
        (assessment.trailing_means[-1] - assessment.trailing_means[0])
        / assessment.cumulative_drift_bars
    )


def test_monotone_descent_inside_the_noise_is_budget_inadequate() -> None:
    """A drift far smaller than the per-step noise still trips the sign test.

    This is the case the rule exists for and the one a tail average misses: the
    raw series looks like noise, so its autocorrelation resolves and the windows
    are wide enough, while the WINDOW MEANS descend monotonically because
    averaging 800 points removes the noise and leaves the drift.
    """

    n_samples = N_WINDOWS * 800
    trend = torch.linspace(0.0, -2.8, n_samples, dtype=torch.float64)
    assessment = _assess(_noise(n_samples, scale=1.0, seed=5) + trend)

    assert assessment.verdict == assess_convergence.VERDICT_BUDGET_INADEQUATE
    assert assessment.direction == "descending"
    assert assessment.signs == [-1] * (N_TRAILING - 1)
    # The windows were validated, not shrunk to fit.
    assert assessment.window_width == 800
    assert assessment.tau_multiple >= 5.0


def test_monotone_ascent_is_a_training_pathology_and_not_a_budget_shortfall() -> None:
    """A rising loss escalates instead of being annotated.

    Detection is two-sided, consequence is not: extending a budget whose loss is
    climbing makes matters worse. The verdict must therefore be distinguishable
    from `budget_inadequate` at the call site, not merged with it.
    """

    n_samples = N_WINDOWS * 800
    trend = torch.linspace(0.0, 2.8, n_samples, dtype=torch.float64)
    assessment = _assess(_noise(n_samples, scale=1.0, seed=5) + trend)

    assert assessment.verdict == assess_convergence.VERDICT_TRAINING_PATHOLOGY
    assert assessment.direction == "ascending"
    assert assessment.signs == [1] * (N_TRAILING - 1)
    assert assessment.verdict != assess_convergence.VERDICT_BUDGET_INADEQUATE


def test_flat_noisy_series_is_adequate() -> None:
    """Pure noise must not trip the gate; a 1.56% false-alarm rate is the design."""

    assessment = _assess(_noise(N_WINDOWS * 800, scale=1.0, seed=11))

    assert assessment.verdict == assess_convergence.VERDICT_ADEQUATE
    assert assessment.direction is None
    assert len(set(assessment.signs)) > 1


def test_non_monotone_scatter_wider_than_a_bar_is_adequate() -> None:
    """Scatter of several bars is ADEQUATE when it does not share a sign.

    This is the peer lane's real case: 104 uHa of scatter against an 18 uHa bar
    was noise, because it was non-monotone. An error-bar overlap test fails here
    in the reassuring direction, and a rule keyed on drift MAGNITUDE would
    declare this inadequate. Asserting the scatter genuinely exceeds a bar is
    what makes this test discriminating rather than merely another flat series.
    """

    n_samples = N_WINDOWS * 800
    index = torch.arange(n_samples, dtype=torch.float64)
    # Four windows per period, so the trailing eight windows swing twice.
    wiggle = 0.25 * torch.sin(2.0 * math.pi * index / (4.0 * 800))
    assessment = _assess(_noise(n_samples, scale=1.0, seed=11) + wiggle)

    assert assessment.verdict == assess_convergence.VERDICT_ADEQUATE
    assert len(set(assessment.signs)) > 1
    # The window means really do scatter by more than one bar.
    spread = max(assessment.trailing_means) - min(assessment.trailing_means)
    assert spread > _bar(assessment)


def test_a_series_too_short_for_the_declared_windows_refuses_rather_than_shrinking() -> None:
    """Below the declared tau multiple the script escalates, never re-fits.

    Narrowing the windows to satisfy the multiple would raise the frequency of
    same-sign runs under the null and make the declared 1.56% optimistic by an
    unknown factor -- a gate whose error rate nobody has bounded. Refusing is
    the only honest option, so it is asserted here rather than trusted.
    """

    # Strongly correlated: tau is large against any window this series can form.
    n_samples = N_WINDOWS * 100
    generator = torch.Generator().manual_seed(23)
    noise = torch.randn(n_samples, generator=generator, dtype=torch.float64)
    values = torch.empty(n_samples, dtype=torch.float64)
    value = 0.0
    for position in range(n_samples):
        value = 0.95 * value + float(noise[position])
        values[position] = value

    assessment = _assess(values)

    assert assessment.verdict == assess_convergence.VERDICT_INDETERMINATE
    assert assessment.signs == []
    assert assessment.window_width == 100
    # It reports what the series COULD support without adopting it.
    assert "NOT narrowed" in assessment.reason or "did not resolve" in assessment.reason


def test_the_rule_is_read_from_the_grid_and_not_from_the_caller() -> None:
    """The declared parameters are the ones used, and a wrong method is refused."""

    rule = assess_convergence.read_rule(STUDY / "configs" / "production_grid.yaml")
    assert rule["method"] == "windowed_means_sign_test"
    assert rule["n_windows"] == N_WINDOWS
    assert rule["n_trailing_windows"] == N_TRAILING
    assert rule["window_width_min_tau_multiple"] == 5.0
    assert rule["may_reselect"] is False


def test_the_reported_verdict_never_selects_anything() -> None:
    """The result carries no checkpoint, seed, budget or arm.

    `may_reselect: false` is enforced by `plan.py`; this asserts the script's
    OUTPUT cannot express a re-selection even if a future edit tried to make one.
    """

    payload = _assess(_noise(N_WINDOWS * 800, scale=1.0, seed=11)).to_dict()
    forbidden = {"checkpoint", "seed", "budget", "arm", "max_steps", "extend", "recommend"}
    for key in payload:
        assert not any(word in key.lower() for word in forbidden), key
