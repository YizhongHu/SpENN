"""Equivariance infrastructure: traceable maps and passive trace recording."""

from spenn.equivariance.map import EquivariantMap
from spenn.equivariance.trace import (
    EquivarianceTrace,
    EquivarianceTraceEntry,
    EquivarianceTraceWarning,
    trace_equivariant,
)

__all__ = [
    "EquivariantMap",
    "EquivarianceTrace",
    "EquivarianceTraceEntry",
    "EquivarianceTraceWarning",
    "trace_equivariant",
]
