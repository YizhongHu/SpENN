"""Equivariance infrastructure: traceable maps, trace recording, runtime checks."""

from spenn.equivariance.checks import (
    EquivarianceCheckResult,
    FullModelEquivarianceChecker,
    RuntimeEquivarianceChecker,
    TraceEquivarianceChecker,
)
from spenn.equivariance.map import EquivariantMap
from spenn.equivariance.trace import (
    EquivarianceTrace,
    EquivarianceTraceEntry,
    EquivarianceTraceWarning,
    trace_equivariant,
)

__all__ = [
    "EquivariantMap",
    "EquivarianceCheckResult",
    "EquivarianceTrace",
    "EquivarianceTraceEntry",
    "EquivarianceTraceWarning",
    "FullModelEquivarianceChecker",
    "RuntimeEquivarianceChecker",
    "TraceEquivarianceChecker",
    "trace_equivariant",
]
