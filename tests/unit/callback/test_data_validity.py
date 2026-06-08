"""Tests for the DataValidity runtime-check callback."""

from __future__ import annotations

import pytest
import torch

from spenn.callback import DataValidity
from spenn.data.batch import WavefunctionOutput
from tests.unit.callback.support import FakeState, RecordingContext, step_event


def _finite_state() -> FakeState:
    return FakeState(
        step=1,
        batch={"positions": torch.zeros(4, 2, 3, dtype=torch.float64)},
        local_energy=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
        loss=torch.tensor(1.5, dtype=torch.float64),
        wavefunction_output=WavefunctionOutput(
            logabs=torch.tensor([-1.0, -2.0, -3.0, -4.0], dtype=torch.float64),
            sign=torch.ones(4, dtype=torch.float64),
        ),
    )


def _handle(callback: DataValidity, state: FakeState) -> RecordingContext:
    context = RecordingContext()
    callback.handle(step_event(context, state))
    return context


def test_passes_on_finite_state() -> None:
    context = _handle(DataValidity(["step_end"], fail_fast=True), _finite_state())

    metrics = context.latest("checks/data_validity")
    assert metrics["passed"] is True
    assert metrics["local_energy_nonfinite_fraction"] == 0.0
    assert metrics["local_energy_finite_count"] == 4
    assert metrics["local_energy_total_count"] == 4
    assert metrics["logabs_nonfinite_fraction"] == 0.0
    assert metrics["logabs_finite_count"] == 4
    assert metrics["logabs_total_count"] == 4
    assert metrics["loss_is_finite"] is True


def test_counts_disambiguate_empty_from_all_nonfinite() -> None:
    all_nan = _finite_state()
    all_nan.local_energy = torch.full((4,), float("nan"), dtype=torch.float64)
    empty = _finite_state()
    empty.local_energy = torch.empty(0, dtype=torch.float64)

    nan_metrics = _handle(DataValidity(["step_end"], fail_fast=False), all_nan).latest("checks/data_validity")
    empty_metrics = _handle(DataValidity(["step_end"], fail_fast=False), empty).latest("checks/data_validity")

    # Both share fraction 1.0 but the counts tell them apart.
    assert nan_metrics["local_energy_nonfinite_fraction"] == 1.0
    assert empty_metrics["local_energy_nonfinite_fraction"] == 1.0
    assert (nan_metrics["local_energy_finite_count"], nan_metrics["local_energy_total_count"]) == (0, 4)
    assert (empty_metrics["local_energy_finite_count"], empty_metrics["local_energy_total_count"]) == (0, 0)


def test_empty_batch_tensor_counts_as_invalid() -> None:
    state = _finite_state()
    state.batch = {"positions": torch.empty(0, dtype=torch.float64)}

    metrics = _handle(DataValidity(["step_end"], fail_fast=False), state).latest("checks/data_validity")

    assert metrics["batch_nonfinite_tensor_count"] == 1
    assert metrics["passed"] is False


def test_fails_on_nan_local_energy() -> None:
    state = _finite_state()
    state.local_energy = torch.tensor([1.0, float("nan"), 3.0, 4.0], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="local_energy_nonfinite_fraction"):
        _handle(DataValidity(["step_end"], fail_fast=True), state)


def test_fails_on_inf_logabs() -> None:
    state = _finite_state()
    state.wavefunction_output = WavefunctionOutput(
        logabs=torch.tensor([0.0, float("inf"), 0.0, 0.0], dtype=torch.float64),
        sign=torch.ones(4, dtype=torch.float64),
    )

    with pytest.raises(RuntimeError, match="logabs_nonfinite_fraction"):
        _handle(DataValidity(["step_end"], fail_fast=True), state)


def test_fails_on_nonfinite_loss() -> None:
    state = _finite_state()
    state.loss = torch.tensor(float("nan"), dtype=torch.float64)

    with pytest.raises(RuntimeError, match="loss is not finite"):
        _handle(DataValidity(["step_end"], fail_fast=True), state)


def test_strict_sign_values_catches_invalid_sign() -> None:
    state = _finite_state()
    state.wavefunction_output = WavefunctionOutput(
        logabs=torch.zeros(4, dtype=torch.float64),
        sign=torch.tensor([1.0, 0.5, -1.0, 1.0], dtype=torch.float64),
    )

    context = _handle(DataValidity(["step_end"], fail_fast=False, strict_sign_values=True), state)

    metrics = context.latest("checks/data_validity")
    assert metrics["sign_invalid_fraction"] == pytest.approx(0.25)
    assert metrics["passed"] is False


def test_logs_nonfinite_fractions_without_raising_when_not_fail_fast() -> None:
    state = _finite_state()
    state.local_energy = torch.tensor([1.0, float("inf"), float("nan"), 4.0], dtype=torch.float64)

    context = _handle(DataValidity(["step_end"], fail_fast=False), state)

    metrics = context.latest("checks/data_validity")
    assert metrics["local_energy_nonfinite_fraction"] == pytest.approx(0.5)
    assert metrics["passed"] is False


def test_fails_on_nonfinite_batch_tensor() -> None:
    state = _finite_state()
    state.batch = {"positions": torch.tensor([[float("nan")]], dtype=torch.float64)}

    context = _handle(DataValidity(["step_end"], fail_fast=False), state)

    metrics = context.latest("checks/data_validity")
    assert metrics["batch_nonfinite_tensor_count"] == 1
    assert metrics["passed"] is False
