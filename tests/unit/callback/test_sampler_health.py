"""Tests for the SamplerHealth runtime-check callback."""

from __future__ import annotations

import pytest

from spenn.callback import SamplerHealth
from tests.unit.callback.support import FakeState, RecordingContext, step_event


def _handle(callback: SamplerHealth, state: FakeState) -> RecordingContext:
    context = RecordingContext()
    callback.handle(step_event(context, state))
    return context


def test_logs_available_sampler_stats() -> None:
    state = FakeState(sampler_stats={"acceptance_rate": 0.4, "n_walkers": 256, "n_steps": 10, "burn_in": 20})

    metrics = _handle(SamplerHealth(["step_end"]), state).latest("checks/sampler")

    assert metrics["acceptance_rate"] == pytest.approx(0.4)
    assert metrics["n_walkers"] == 256
    assert metrics["n_steps"] == 10
    assert metrics["burn_in"] == 20
    assert metrics["passed"] is True


def test_does_not_require_all_sampler_stats() -> None:
    state = FakeState(sampler_stats={"acceptance_rate": 0.4})

    metrics = _handle(SamplerHealth(["step_end"]), state).latest("checks/sampler")

    assert metrics["acceptance_rate"] == pytest.approx(0.4)
    assert "n_walkers" not in metrics
    assert metrics["passed"] is True


def test_falls_back_to_sampler_prefixed_metrics() -> None:
    state = FakeState(metrics={"sampler.acceptance_rate": 0.3, "sampler.n_walkers": 128, "loss": 1.0})

    metrics = _handle(SamplerHealth(["step_end"]), state).latest("checks/sampler")

    assert metrics["acceptance_rate"] == pytest.approx(0.3)
    assert metrics["n_walkers"] == 128


def test_acceptance_bounds_set_passed_false_without_raising() -> None:
    state = FakeState(sampler_stats={"acceptance_rate": 0.1})

    metrics = _handle(
        SamplerHealth(["step_end"], fail_fast=False, min_acceptance_rate=0.2), state
    ).latest("checks/sampler")

    assert metrics["passed"] is False


def test_fail_fast_raises_when_acceptance_out_of_bounds() -> None:
    state = FakeState(sampler_stats={"acceptance_rate": 0.95})

    with pytest.raises(RuntimeError, match="acceptance_rate"):
        _handle(SamplerHealth(["step_end"], fail_fast=True, max_acceptance_rate=0.9), state)
