"""Composable evaluation framework."""

from __future__ import annotations

from spenn.evaluation.bundle import (
    EvaluationBundle,
    GeneratedConfigurations,
    LocalEnergyValues,
    WavefunctionValues,
)
from spenn.evaluation.calculators import (
    LocalEnergyCalculator,
    WavefunctionCalculator,
)
from spenn.evaluation.evaluator import Evaluator
from spenn.evaluation.generators import MCMCGenerator
from spenn.evaluation.protocols import EvaluationContext
from spenn.evaluation.results import ArtifactRecord, EvaluationFailure, EvaluationResult, SummaryResult, TaskResult
from spenn.evaluation.summaries import (
    HamiltonianTermSummary,
    LocalEnergySummary,
    ReferenceEnergySummary,
    SampledRecordWriter,
    SamplerStatsSummary,
)
from spenn.evaluation.task import EvaluationTask

__all__ = [
    "ArtifactRecord",
    "EvaluationBundle",
    "EvaluationContext",
    "EvaluationFailure",
    "EvaluationResult",
    "EvaluationTask",
    "Evaluator",
    "GeneratedConfigurations",
    "HamiltonianTermSummary",
    "LocalEnergyCalculator",
    "LocalEnergyValues",
    "LocalEnergySummary",
    "MCMCGenerator",
    "ReferenceEnergySummary",
    "SampledRecordWriter",
    "SamplerStatsSummary",
    "SummaryResult",
    "TaskResult",
    "WavefunctionCalculator",
    "WavefunctionValues",
]
