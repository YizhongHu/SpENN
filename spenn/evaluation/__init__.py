"""Composable evaluation framework."""

from __future__ import annotations

from spenn.evaluation.bundle import (
    DerivativeValues,
    EvaluationBundle,
    GeneratedConfigurations,
    LocalEnergyValues,
    WavefunctionValues,
)
from spenn.evaluation.calculators import (
    LocalEnergyCalculator,
    RadialLogAbsDerivativeCalculator,
    WavefunctionCalculator,
)
from spenn.evaluation.evaluator import Evaluator
from spenn.evaluation.generators import (
    CuspGridGenerator,
    HookeOrbitalGenerator,
    MCMCGenerator,
    StratifiedGeometryGenerator,
    TailGridGenerator,
)
from spenn.evaluation.protocols import EvaluationContext
from spenn.evaluation.results import ArtifactRecord, EvaluationFailure, EvaluationResult, SummaryResult, TaskResult
from spenn.evaluation.summaries import (
    CoalescenceDivergenceSummary,
    HamiltonianTermSummary,
    LocalEnergySummary,
    OppositeSpinCuspSummary,
    PathologyCountSummary,
    ReferenceEnergySummary,
    SampledRecordWriter,
    SamplerStatsSummary,
    TailStabilitySummary,
)
from spenn.evaluation.task import EvaluationTask

__all__ = [
    "ArtifactRecord",
    "CoalescenceDivergenceSummary",
    "CuspGridGenerator",
    "DerivativeValues",
    "EvaluationBundle",
    "EvaluationContext",
    "EvaluationFailure",
    "EvaluationResult",
    "EvaluationTask",
    "Evaluator",
    "GeneratedConfigurations",
    "HamiltonianTermSummary",
    "HookeOrbitalGenerator",
    "LocalEnergyCalculator",
    "LocalEnergyValues",
    "LocalEnergySummary",
    "MCMCGenerator",
    "OppositeSpinCuspSummary",
    "PathologyCountSummary",
    "RadialLogAbsDerivativeCalculator",
    "ReferenceEnergySummary",
    "SampledRecordWriter",
    "SamplerStatsSummary",
    "StratifiedGeometryGenerator",
    "SummaryResult",
    "TailGridGenerator",
    "TailStabilitySummary",
    "TaskResult",
    "WavefunctionCalculator",
    "WavefunctionValues",
]
