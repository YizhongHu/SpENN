"""Tests for task-local output directory contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import pytest
from torch import nn

from spenn.evaluation import Evaluator, EvaluationTask
from spenn.evaluation.bundle import EvaluationBundle, GeneratedConfigurations
from spenn.data.batch import ElectronBatch
from spenn.evaluation.protocols import EvaluationContext
from spenn.evaluation.results import SummaryResult


class _NullGenerator:
    name = "null"

    def generate(self, *, model: nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        batch = ElectronBatch(
            positions=torch.zeros(1, 2, 3, dtype=torch.float64),
            spins=torch.tensor([[1.0, -1.0]], dtype=torch.float64),
        )
        return GeneratedConfigurations(batch=batch, metadata={})


class _RecordingOutputDirSummary:
    name = "output_dir_recorder"
    required_fields: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self.recorded_task_output_dir: Path | None = None

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        self.recorded_task_output_dir = context.task_output_dir
        return SummaryResult(metrics={})


def _run_context(run_dir: Path) -> Any:
    ctx = SimpleNamespace()
    ctx.run_dir = run_dir
    ctx.metadata = SimpleNamespace(device=None, dtype=None)
    ctx.log = lambda *a, **kw: None
    return ctx


def test_evaluator_requires_explicit_task_output_dir() -> None:
    with pytest.raises(ValueError, match="output_dir"):
        Evaluator(
            namespace="eval",
            tasks=[
                {
                    "name": "energy",
                    "namespace": "eval/energy",
                    "generator": _NullGenerator(),
                    "calculators": [],
                    "summaries": [],
                }
            ],
        )


def test_task_output_dir_is_respected(tmp_path: Path) -> None:
    explicit_dir = tmp_path / "energy"
    recorder = _RecordingOutputDirSummary()
    evaluator = Evaluator(
        namespace="eval",
        tasks=[
            EvaluationTask(
                name="energy",
                namespace="eval/energy",
                output_dir=explicit_dir,
                generator=_NullGenerator(),
                calculators=[],
                summaries=[recorder],
            )
        ],
    )
    evaluator.evaluate(model=nn.Linear(1, 1), context=_run_context(tmp_path), emit=lambda *a, **kw: None)
    assert recorder.recorded_task_output_dir == explicit_dir


def test_task_output_dir_override_is_respected(tmp_path: Path) -> None:
    custom_dir = tmp_path / "custom_energy_output"
    recorder = _RecordingOutputDirSummary()
    evaluator = Evaluator(
        namespace="eval",
        tasks=[
            EvaluationTask(
                name="energy",
                namespace="eval/energy",
                output_dir=custom_dir,
                generator=_NullGenerator(),
                calculators=[],
                summaries=[recorder],
            )
        ],
    )
    evaluator.evaluate(model=nn.Linear(1, 1), context=_run_context(tmp_path), emit=lambda *a, **kw: None)
    assert recorder.recorded_task_output_dir == custom_dir
