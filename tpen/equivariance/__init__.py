"""Equivariance infrastructure: traceable maps, trace recording, runtime checks."""

from tpen.equivariance.checks import (
    EquivarianceCheckResult,
    FullModelEquivarianceChecker,
    RuntimeEquivarianceChecker,
    TraceEquivarianceChecker,
)
from tpen.equivariance.map import EquivariantMap
from tpen.trace import Trace, TraceEntry, TraceWarning, trace_value

__all__ = [
    "EquivariantMap",
    "EquivarianceCheckResult",
    "FullModelEquivarianceChecker",
    "RuntimeEquivarianceChecker",
    "Trace",
    "TraceEquivarianceChecker",
    "TraceEntry",
    "TraceWarning",
    "trace_value",
]
