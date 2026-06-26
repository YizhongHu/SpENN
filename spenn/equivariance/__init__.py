"""Equivariance infrastructure: traceable maps, trace recording, runtime checks."""

from spenn.equivariance.checks import (
    EquivarianceCheckResult,
    FullModelEquivarianceChecker,
    RuntimeEquivarianceChecker,
    TraceEquivarianceChecker,
)
from spenn.equivariance.map import EquivariantMap
from spenn.trace import Trace, TraceEntry, TraceWarning, trace_value

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
