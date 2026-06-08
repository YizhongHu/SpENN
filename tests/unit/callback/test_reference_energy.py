"""Tests for the ReferenceEnergy callback."""

from __future__ import annotations

import pytest

from spenn.callback import ReferenceEnergy
from tests.unit.callback.support import FakeState, RecordingContext, step_event


def test_logs_reference_energy_metrics() -> None:
    context = RecordingContext()
    state = FakeState(metrics={"energy_mean": 2.5})

    ReferenceEnergy(["step_end"], reference_energy=2.0).handle(step_event(context, state))

    metrics = context.latest("reference")
    assert metrics["reference_energy"] == pytest.approx(2.0)
    assert metrics["energy_error"] == pytest.approx(0.5)
    assert metrics["abs_energy_error"] == pytest.approx(0.5)


def test_uses_configured_source_metric_and_namespace() -> None:
    context = RecordingContext()
    state = FakeState(metrics={"energy_running_mean": 1.5})

    ReferenceEnergy(
        ["step_end"], reference_energy=2.0, source_metric="energy_running_mean", namespace="ref"
    ).handle(step_event(context, state))

    assert context.by_namespace("ref")[-1]["metrics"]["abs_energy_error"] == pytest.approx(0.5)


def test_raises_when_source_metric_missing() -> None:
    state = FakeState(metrics={"loss": 1.0})

    with pytest.raises(KeyError, match="energy_mean"):
        ReferenceEnergy(["step_end"], reference_energy=2.0).handle(step_event(RecordingContext(), state))
