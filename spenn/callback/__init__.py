"""Callback primitives for configured SpENN runs."""

from __future__ import annotations

import torch

from .base import Callback, Event
from .checkpoint import Checkpoint
from .equivariance import RuntimeEquivariance
from .health import DataValidity, GradientStats, SamplerHealth
from .metadata import Metadata
from .snapshot import ConfigSnapshot, ResolvedConfigSnapshot
from .status import Status, configure_terminal_logging
from .timing import DiagnosticTiming, EvaluationTiming, RunTiming, TrainStepTiming

__all__ = [
    "Callback",
    "Checkpoint",
    "ConfigSnapshot",
    "DataValidity",
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
    "configure_terminal_logging",
]
