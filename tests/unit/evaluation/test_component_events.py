"""Tests for evaluator component lifecycle events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from tpen.artifacts import RunContext
from tpen.data.batch import ElectronBatch
from tpen.evaluation import Evaluator, EvaluationTask
from tpen.evaluation.bundle import EvaluationBundle, GeneratedConfigurations
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.results import SummaryResult
from tests.helpers.run_context import make_run_context


class _NullGenerator:
    name = "null"

    def generate(self, *, model: nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        batch = ElectronBatch(
            positions=torch.zeros(1, 2, 3, dtype=torch.float64),
            spins=torch.tensor([[1.0, -1.0]], dtype=torch.float64),
        )
        return GeneratedConfigurations(batch=batch, metadata={})


class _FailingGenerator:
    name = "broken-generator"

    def generate(self, *, model: nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        raise RuntimeError("generator boom")


class _IdentityCalculator:
    name = "identity"

    def calculate(self, *, model: nn.Module | None, bundle: EvaluationBundle, context: EvaluationContext) -> EvaluationBundle:
        return bundle


class _FailingCalculator:
    name = "broken"

    def calculate(self, *, model: nn.Module | None, bundle: EvaluationBundle, context: EvaluationContext) -> EvaluationBundle:
        raise RuntimeError("calculator boom")


class _MetricSummary:
    name = "metric"
    required_fields: frozenset[str] = frozenset()

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        return SummaryResult(metrics={"value": 1.0})


def _run_context(run_dir: Path) -> RunContext:
    """Return a real `RunContext`.

    The evaluator now emits typed occurrences through the context itself, so a
    `types.SimpleNamespace` stand-in no longer suffices: `RunContext.scope`
    needs the occurrence counters and `write_occurrence_artifact` needs the
    artifact manager and the run clock.
    """

    return make_run_context(run_dir)


def _evaluator(tmp_path: Path, calculators: list[object], *, generator: object | None = None) -> Evaluator:
    return Evaluator(
        namespace="eval",
        tasks=[
            EvaluationTask(
                name="energy",
                namespace="eval/energy",
                output_dir=tmp_path / "energy",
                generator=_NullGenerator() if generator is None else generator,
                calculators=calculators,
                summaries=[_MetricSummary()],
            )
        ],
    )


def _emit_recorder(events: list[tuple[str, dict[str, Any]]]):
    def emit(name: str, *, payload: dict[str, Any] | None = None) -> None:
        events.append((name, dict(payload or {})))

    return emit


def test_component_events_bracket_each_component_on_success(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    evaluator = _evaluator(tmp_path, calculators=[_IdentityCalculator()])

    result = evaluator.evaluate(
        model=nn.Linear(1, 1), context=_run_context(tmp_path), emit=_emit_recorder(events)
    )

    assert result.status == "success"
    assert [name for name, _ in events] == [
        "task_start",
        "generator_start",
        "generator_end",
        "calculator_start",
        "calculator_end",
        "summary_start",
        "summary_end",
        "task_end",
    ]
    by_name = dict(events)
    for event_name, component_name in [
        ("generator_start", "null"),
        ("generator_end", "null"),
        ("calculator_start", "identity"),
        ("calculator_end", "identity"),
        ("summary_start", "metric"),
        ("summary_end", "metric"),
    ]:
        payload = by_name[event_name]
        assert payload["task_name"] == "energy"
        assert payload["task_namespace"] == "eval/energy"
        assert payload["component_name"] == component_name
        assert payload["output_dir"] == str(tmp_path / "energy")


def test_calculator_end_fires_before_calculator_failed_on_failure(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    evaluator = _evaluator(tmp_path, calculators=[_FailingCalculator()])

    result = evaluator.evaluate(
        model=nn.Linear(1, 1), context=_run_context(tmp_path), emit=_emit_recorder(events)
    )

    assert result.status == "failed"
    # The default "continue" policy still runs summaries after a calculator
    # failure; calculator_end fires before calculator_failed either way.
    assert [name for name, _ in events] == [
        "task_start",
        "generator_start",
        "generator_end",
        "calculator_start",
        "calculator_end",
        "calculator_failed",
        "summary_start",
        "summary_end",
        "task_failed",
    ]
    by_name = dict(events)
    assert by_name["calculator_end"]["component_name"] == "broken"
    assert by_name["calculator_failed"]["failure"]["error_type"] == "RuntimeError"


def test_generator_failure_emits_task_failed_first(tmp_path: Path) -> None:
    """Pin the generator path's inverted failure order.

    Unlike the calculator and summary paths, a failing generator reports
    ``task_failed`` BEFORE ``generator_failed``: that path has to build the task
    result to return it, and does so before reporting the component. Nothing
    pinned that asymmetry, so a later rewrite could normalise it by accident and
    silently reorder what `FailureLog` and `DiagnosticTiming` observe. Pinning it
    makes keeping it, or changing it, a deliberate choice.
    """

    events: list[tuple[str, dict[str, Any]]] = []
    evaluator = _evaluator(tmp_path, calculators=[_IdentityCalculator()], generator=_FailingGenerator())

    result = evaluator.evaluate(
        model=nn.Linear(1, 1), context=_run_context(tmp_path), emit=_emit_recorder(events)
    )

    assert result.status == "failed"
    assert [name for name, _ in events] == [
        "task_start",
        "generator_start",
        "generator_end",
        "task_failed",
        "generator_failed",
    ]
    by_name = dict(events)
    assert by_name["generator_failed"]["component_name"] == "broken-generator"
    assert by_name["generator_failed"]["failure"]["error_type"] == "RuntimeError"
    # No calculator or summary runs once the generator has failed.
    assert not any(name.startswith(("calculator", "summary")) for name, _ in events)
