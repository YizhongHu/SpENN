"""Tests for the RuntimeEquivariance callback and EquivarianceChecker."""

from __future__ import annotations

import pytest
import torch

from spenn.callback import RuntimeEquivariance
from spenn.data.real import RealFeature, zero_block
from spenn.equivariance import EquivariantMap
from spenn.testing.runtime import CheckResult, EquivarianceChecker
from tests.unit.callback.support import FakeState, RecordingContext, step_event


class FakeChecker:
    """Checker stub returning a preset result and counting calls."""

    def __init__(self, *, name: str = "equivariance", passed: bool = True) -> None:
        self.name = name
        self.passed = passed
        self.metrics = {"max_abs_error": 0.0, "passed": passed}
        self.calls = 0

    def run(self, state) -> CheckResult:
        self.calls += 1
        return CheckResult(name=self.name, passed=self.passed, metrics=self.metrics)


def _feature() -> RealFeature:
    # Last axis is the particle index (3 particles), channels = 2.
    return RealFeature(
        [
            zero_block(dtype=torch.float64),
            torch.arange(1 * 2 * 3, dtype=torch.float64).reshape(1, 2, 3),
            torch.arange(1 * 2 * 3 * 3, dtype=torch.float64).reshape(1, 2, 3, 3),
        ]
    )


class IdentityMap(EquivariantMap):
    def forward_impl(self, x: RealFeature) -> RealFeature:
        return x.clone()


class LabelWeightedMap(EquivariantMap):
    def forward_impl(self, x: RealFeature) -> RealFeature:
        weights = torch.tensor([1.0, 2.0, 4.0], dtype=x.blocks[1].dtype).reshape(1, 1, 3)
        return RealFeature([x.blocks[0].clone(), x.blocks[1] * weights, x.blocks[2].clone()])


class LabelBiasMap(EquivariantMap):
    def forward_impl(self, x: RealFeature) -> RealFeature:
        n = x.blocks[1].shape[-1]
        bias = torch.arange(n, dtype=x.blocks[1].dtype).reshape(1, 1, n)
        return RealFeature([x.blocks[0].clone(), x.blocks[1] + bias, x.blocks[2].clone()])


# --- RuntimeEquivariance callback (fake checker) ---


def test_callback_calls_checker_and_logs_metrics() -> None:
    checker = FakeChecker(passed=True)
    context = RecordingContext()

    RuntimeEquivariance(["step_end"], checker=checker).handle(step_event(context, FakeState()))

    assert checker.calls == 1
    record = context.by_namespace("checks/equivariance")[-1]
    assert record["metrics"] == checker.metrics


def test_callback_raises_when_failed_and_fail_fast() -> None:
    checker = FakeChecker(passed=False)

    with pytest.raises(RuntimeError, match="equivariance"):
        RuntimeEquivariance(["step_end"], checker=checker, fail_fast=True).handle(
            step_event(RecordingContext(), FakeState())
        )


def test_callback_does_not_raise_when_not_fail_fast() -> None:
    checker = FakeChecker(passed=False)
    context = RecordingContext()

    RuntimeEquivariance(["step_end"], checker=checker, fail_fast=False).handle(
        step_event(context, FakeState())
    )

    assert context.latest("checks/equivariance")["passed"] is False


def test_callback_runs_checker_only_on_scheduled_steps() -> None:
    checker = FakeChecker()
    callback = RuntimeEquivariance(["step_end"], checker=checker, every_n_steps=2)

    for step in range(0, 5):
        callback.handle(step_event(RecordingContext(), FakeState(step=step), step=step))

    assert checker.calls == 3  # steps 0, 2, 4


# --- EquivarianceChecker (real toy maps) ---


def test_equivariance_checker_passes_on_equivariant_map() -> None:
    result = EquivarianceChecker(n_permutations=2, seed=0).run(
        FakeState(model=IdentityMap(), batch=_feature())
    )

    assert result.name == "equivariance"
    assert result.passed is True
    assert result.metrics["max_abs_error"] == pytest.approx(0.0, abs=1e-9)
    assert result.metrics["n_permutations"] >= 1


@pytest.mark.parametrize("module_cls", [LabelWeightedMap, LabelBiasMap])
def test_equivariance_checker_catches_non_equivariant_map(module_cls) -> None:
    result = EquivarianceChecker(n_permutations=2, seed=0).run(
        FakeState(model=module_cls(), batch=_feature())
    )

    assert result.passed is False
    assert result.metrics["max_abs_error"] > 0.0


def test_equivariance_checker_passes_trivially_without_particle_count() -> None:
    result = EquivarianceChecker(n_permutations=2, seed=0).run(
        FakeState(model=IdentityMap(), batch=None)
    )

    assert result.passed is True
    assert result.metrics["n_permutations"] == 0
