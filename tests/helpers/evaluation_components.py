"""Minimal evaluation components for driving a real `Evaluator` in tests.

Three test modules need the same tiny generator/calculator/summary trio plus
failing variants of each, so they live here rather than being redefined next to
whichever test needed one first. Each declares ``name`` explicitly, so the
component names that reach metric keys and failure records cannot drift with a
class rename.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from tpen.data.batch import ElectronBatch
from tpen.evaluation import EvaluationTask, Evaluator
from tpen.evaluation.bundle import EvaluationBundle, GeneratedConfigurations
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.results import ArtifactRecord, SummaryResult


class NullGenerator:
    """Produce one fixed two-electron configuration."""

    name = "null"

    def generate(self, *, model: nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        batch = ElectronBatch(
            positions=torch.zeros(1, 2, 3, dtype=torch.float64),
            spins=torch.tensor([[1.0, -1.0]], dtype=torch.float64),
        )
        return GeneratedConfigurations(batch=batch, metadata={})


class FailingGenerator:
    """Raise instead of generating."""

    name = "broken-generator"

    def generate(self, *, model: nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        raise RuntimeError("generator boom")


class IdentityCalculator:
    """Pass the bundle straight through."""

    name = "identity"

    def calculate(
        self, *, model: nn.Module | None, bundle: EvaluationBundle, context: EvaluationContext
    ) -> EvaluationBundle:
        return bundle


class FailingCalculator:
    """Raise instead of calculating."""

    name = "broken"

    def calculate(
        self, *, model: nn.Module | None, bundle: EvaluationBundle, context: EvaluationContext
    ) -> EvaluationBundle:
        raise RuntimeError("calculator boom")


class MetricSummary:
    """Emit one metric, and optionally one artifact record."""

    required_fields: frozenset[str] = frozenset()

    def __init__(self, *, name: str = "metric", artifact: ArtifactRecord | None = None) -> None:
        self.name = name
        self._artifact = artifact

    def summarize(
        self, *, bundle: EvaluationBundle, context: EvaluationContext, namespace: str
    ) -> SummaryResult:
        artifacts = () if self._artifact is None else (self._artifact,)
        return SummaryResult(metrics={"value": 1.0}, artifacts=artifacts)


class MissingFieldSummary:
    """Declare a bundle dependency the evaluator can never satisfy.

    Drives the partial-failure path, which reports a `ComponentFailed` without
    ever opening a summary scope.
    """

    name = "needs-missing"
    required_fields: frozenset[str] = frozenset({"local_energy"})

    def summarize(
        self, *, bundle: EvaluationBundle, context: EvaluationContext, namespace: str
    ) -> SummaryResult:  # pragma: no cover - never reached
        raise AssertionError("summary ran despite a missing dependency")


def single_task_evaluator(
    output_root: Path,
    *,
    name: str = "energy",
    generator: object | None = None,
    calculators: list[object] | None = None,
    summaries: list[object] | None = None,
) -> Evaluator:
    """Return an `Evaluator` running exactly one task under ``output_root``."""

    return Evaluator(
        namespace="eval",
        tasks=[
            EvaluationTask(
                name=name,
                namespace=f"eval/{name}",
                output_dir=output_root / name,
                generator=NullGenerator() if generator is None else generator,
                calculators=[IdentityCalculator()] if calculators is None else calculators,
                summaries=[MetricSummary()] if summaries is None else summaries,
            )
        ],
    )


def multi_task_evaluator(output_root: Path, *names: str) -> Evaluator:
    """Return an `Evaluator` running one trivial task per name, in order."""

    return Evaluator(
        namespace="eval",
        tasks=[
            EvaluationTask(
                name=name,
                namespace=f"eval/{name}",
                output_dir=output_root / name,
                generator=NullGenerator(),
                calculators=[IdentityCalculator()],
                summaries=[MetricSummary()],
            )
            for name in names
        ],
    )


__all__ = [
    "FailingCalculator",
    "FailingGenerator",
    "IdentityCalculator",
    "MetricSummary",
    "MissingFieldSummary",
    "NullGenerator",
    "multi_task_evaluator",
    "single_task_evaluator",
]
