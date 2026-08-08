"""Tests for the SamplerHealth runtime-check callback."""

from __future__ import annotations

import pytest

from tpen.callback import SamplerHealth
from tests.unit.callback.support import FakeState, RecordingContext, make_sampler_stats, step_event


def _handle(callback: SamplerHealth, state: FakeState) -> RecordingContext:
    context = RecordingContext()
    callback.handle(step_event(context, state))
    return context


def test_logs_available_sampler_stats() -> None:
    state = FakeState(
        sampler_stats=make_sampler_stats(
            acceptance_rate=0.4, n_walkers=256, n_steps=10, burn_in=20
        )
    )

    metrics = _handle(SamplerHealth(["step_end"]), state).latest("checks/sampler")

    assert metrics["acceptance_rate"] == pytest.approx(0.4)
    assert metrics["n_walkers"] == 256
    assert metrics["n_steps"] == 10
    assert metrics["burn_in"] == 20
    assert metrics["passed"] is True


def test_check_keys_are_the_durable_fixed_set() -> None:
    state = FakeState(
        sampler_stats=make_sampler_stats(geometry={"radius_mean": 1.5, "n_electrons": 2}, seed=7)
    )

    metrics = _handle(SamplerHealth(["step_end"]), state).latest("checks/sampler")

    # Geometry and seed belong to */sampler, never to the fixed-width check.
    assert list(metrics) == ["acceptance_rate", "n_walkers", "n_steps", "burn_in", "passed"]


def test_missing_sampler_stats_still_reports_passed() -> None:
    state = FakeState(sampler_stats=None)

    metrics = _handle(SamplerHealth(["step_end"]), state).latest("checks/sampler")

    assert metrics == {"passed": True}


def test_ignores_sampler_prefixed_metric_keys() -> None:
    state = FakeState(metrics={"sampler.acceptance_rate": 0.3, "sampler.n_walkers": 128, "loss": 1.0})

    metrics = _handle(SamplerHealth(["step_end"]), state).latest("checks/sampler")

    assert "acceptance_rate" not in metrics
    assert "n_walkers" not in metrics
    assert metrics["passed"] is True


def test_acceptance_bounds_set_passed_false_without_raising() -> None:
    state = FakeState(sampler_stats=make_sampler_stats(acceptance_rate=0.1))

    metrics = _handle(
        SamplerHealth(["step_end"], fail_fast=False, min_acceptance_rate=0.2), state
    ).latest("checks/sampler")

    assert metrics["passed"] is False


def test_fail_fast_raises_when_acceptance_out_of_bounds() -> None:
    state = FakeState(sampler_stats=make_sampler_stats(acceptance_rate=0.95))

    with pytest.raises(RuntimeError, match="acceptance_rate"):
        _handle(SamplerHealth(["step_end"], fail_fast=True, max_acceptance_rate=0.9), state)
