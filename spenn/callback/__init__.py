"""Callback primitives for configured SpENN runs."""

from __future__ import annotations

from .base import Callback, Event
from .checkpoint import Checkpoint, load_model_checkpoint
from .equivariance import RuntimeEquivariance
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
    if name == "EvaluationTiming":
        from .timing import EvaluationTiming

        return EvaluationTiming
    if name == "RunTiming":
        from .timing import RunTiming

        return RunTiming
    if name == "TrainStepTiming":
        from .timing import TrainStepTiming

        return TrainStepTiming
    if name == "Validation":
        from .validation import Validation

        return Validation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Callback",
    "Checkpoint",
    "ConfigSnapshot",
    "DataIntegrity",
    "DiagnosticTiming",
    "Event",
    "EvaluationTiming",
    "GradientStats",
    "Metadata",
    "ResolvedConfigSnapshot",
    "RunTiming",
    "RuntimeEquivariance",
    "SamplerHealth",
    "Status",
    "TrainStepTiming",
    "Validation",
    "configure_terminal_logging",
    "load_model_checkpoint",
]
