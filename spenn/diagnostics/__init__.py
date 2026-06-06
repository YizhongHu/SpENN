"""Reusable metric and plot-data diagnostics."""

from spenn.diagnostics.base import Diagnostic, DiagnosticContext, DiagnosticResult
from spenn.diagnostics.wavefunction import (
    CuspSlopeDiagnostic,
    ExchangeSymmetryDiagnostic,
    HistogramDiagnostic,
    RadialCutDiagnostic,
    RadialLogAbsComparison,
)


class Exchange(Diagnostic):
    """Placeholder config target for future exchange diagnostics."""

    name = "exchange"


class Cusp(Diagnostic):
    """Placeholder config target for future cusp diagnostics."""

    name = "cusp"


class Energy(Diagnostic):
    """Placeholder config target for future energy diagnostics."""

    name = "energy"


__all__ = [
    "Cusp",
    "CuspSlopeDiagnostic",
    "Diagnostic",
    "DiagnosticContext",
    "DiagnosticResult",
    "Energy",
    "Exchange",
    "ExchangeSymmetryDiagnostic",
    "HistogramDiagnostic",
    "RadialCutDiagnostic",
    "RadialLogAbsComparison",
]
