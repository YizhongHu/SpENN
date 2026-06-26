"""Tests for the RuntimeEquivariance callback (multi-checker, artifact writing)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spenn.callback import RuntimeEquivariance
from spenn.equivariance.checks import EquivarianceCheckResult
from tests.unit.callback.support import FakeState, RecordingContext, step_event


class FakePassingChecker:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, state) -> EquivarianceCheckResult:
        self.calls += 1
        return EquivarianceCheckResult(passed=True, metrics={"max_abs_error": 0.0})


class FakeFailingChecker:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, state) -> EquivarianceCheckResult:
        self.calls += 1
        return EquivarianceCheckResult(
            passed=False,
            metrics={"max_abs_error": 1.0},
            failures=["boom"],
            artifact={"checker_class": "FakeFailingChecker", "failures": ["boom"]},
        )


def _handle(callback: RuntimeEquivariance, context: RecordingContext, step: int = 1) -> None:
    callback.handle(step_event(context, FakeState(step=step), step=step))


def test_runs_all_checkers_and_logs_class_derived_namespaces() -> None:
    passing, failing = FakePassingChecker(), FakeFailingChecker()
    context = RecordingContext()

    _handle(RuntimeEquivariance(["step_end"], checkers=[passing, failing], fail_fast=False), context)

    assert passing.calls == 1 and failing.calls == 1
    namespaces = {record["namespace"] for record in context.records}
    assert namespaces == {"checks/equivariance/fake_passing", "checks/equivariance/fake_failing"}
    passing_metrics = context.latest("checks/equivariance/fake_passing")
    assert passing_metrics["passed"] is True
    assert passing_metrics["checker_class"] == "FakePassingChecker"


def test_duplicate_names_warn_and_disambiguate_without_failing() -> None:
    context = RecordingContext()
    with pytest.warns(UserWarning, match="duplicate checker name 'fake_passing'"):
        callback = RuntimeEquivariance(
            ["step_end"], checkers=[FakePassingChecker(), FakePassingChecker()], fail_fast=False
        )

    _handle(callback, context)

    namespaces = {record["namespace"] for record in context.records}
    assert namespaces == {"checks/equivariance/fake_passing", "checks/equivariance/fake_passing_1"}


def test_writes_artifact_and_adds_artifact_path_metric(tmp_path: Path) -> None:
    context = RecordingContext()
    callback = RuntimeEquivariance(
        ["step_end"], checkers=[FakeFailingChecker()], fail_fast=False, artifact_dir=tmp_path
    )

    _handle(callback, context, step=12)

    metrics = context.latest("checks/equivariance/fake_failing")
    artifact_path = Path(metrics["artifact_path"])
    assert artifact_path == tmp_path / "fake_failing" / "step_000012" / "failure.json"
    assert artifact_path.exists()
    assert json.loads(artifact_path.read_text())["checker_class"] == "FakeFailingChecker"


def test_no_artifact_written_without_artifact_dir() -> None:
    context = RecordingContext()
    callback = RuntimeEquivariance(["step_end"], checkers=[FakeFailingChecker()], fail_fast=False)

    _handle(callback, context)

    assert "artifact_path" not in context.latest("checks/equivariance/fake_failing")


def test_fail_fast_raises_on_failure() -> None:
    callback = RuntimeEquivariance(["step_end"], checkers=[FakeFailingChecker()], fail_fast=True)

    with pytest.raises(RuntimeError, match="fake_failing"):
        _handle(callback, RecordingContext())


def test_no_raise_when_not_fail_fast() -> None:
    context = RecordingContext()

    _handle(RuntimeEquivariance(["step_end"], checkers=[FakeFailingChecker()], fail_fast=False), context)

    assert context.latest("checks/equivariance/fake_failing")["passed"] is False


def test_checkers_run_only_on_scheduled_steps() -> None:
    checker = FakePassingChecker()
    callback = RuntimeEquivariance(["step_end"], checkers=[checker], every_n_steps=2, fail_fast=False)

    for step in range(0, 5):
        callback.handle(step_event(RecordingContext(), FakeState(step=step), step=step))

    assert checker.calls == 3  # steps 0, 2, 4
