"""Callback primitives for configured TPEN runs."""

from __future__ import annotations

from .base import Callback, Event, StatefulCallback
from .cadence import Cadence, CadenceGate, SubscriptionGroup
from .checkpoint import Checkpoint
from .equivariance import RuntimeEquivariance
from .evaluation import ArtifactIndex, FailureLog
from .metadata import Metadata
from .snapshot import ConfigSnapshot, ResolvedConfigSnapshot
from .status import Status, configure_terminal_logging


def __getattr__(name: str) -> object:
    """Load torch-dependent callback classes only when they are requested."""

    if name == "DataIntegrity":
        from .health import DataIntegrity

        return DataIntegrity
    if name == "GradientStats":
        from .health import GradientStats

        return GradientStats
    if name == "SamplerHealth":
        from .health import SamplerHealth

        return SamplerHealth
    if name == "DiagnosticTiming":
        from .timing import DiagnosticTiming

        return DiagnosticTiming
    if name == "EvaluationComponentTiming":
        from .timing import EvaluationComponentTiming

        return EvaluationComponentTiming
    if name == "EvaluationTiming":
        from .timing import EvaluationTiming

        return EvaluationTiming
    if name == "ResourceUsage":
        from .resource_usage import ResourceUsage

        return ResourceUsage
    if name == "RunTiming":
        from .timing import RunTiming

        return RunTiming
    if name == "TrainPhaseTiming":
        from .timing import TrainPhaseTiming

        return TrainPhaseTiming
    if name == "TrainStepTiming":
        from .timing import TrainStepTiming

        return TrainStepTiming
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ArtifactIndex",
    "Cadence",
    "CadenceGate",
    "Callback",
    "Checkpoint",
    "ConfigSnapshot",
    "DataIntegrity",
    "DiagnosticTiming",
    "EvaluationComponentTiming",
    "EvaluationTiming",
    "Event",
    "FailureLog",
    "GradientStats",
    "Metadata",
    "ResolvedConfigSnapshot",
    "ResourceUsage",
    "RunTiming",
    "RuntimeEquivariance",
    "SamplerHealth",
    "StatefulCallback",
    "Status",
    "SubscriptionGroup",
    "TrainPhaseTiming",
    "TrainStepTiming",
    "configure_terminal_logging",
]
