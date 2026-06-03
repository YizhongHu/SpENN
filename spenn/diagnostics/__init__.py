"""Reusable metric and plot-data diagnostics."""

from spenn.diagnostics.base import DiagnosticContext, DiagnosticResult
from spenn.diagnostics.wavefunction import (
    ParticleAntisymmetryDiagnostic,
    CuspSlopeDiagnostic,
    ExchangeSymmetryDiagnostic,
    HistogramDiagnostic,
    PairDistanceHistogramDiagnostic,
    RadialCutDiagnostic,
    RadialDensityDiagnostic,
    RadialLogAbsComparison,
    SpinResolvedCuspSlopeDiagnostic,
    all_pair_distances,
)

__all__ = [
    "all_pair_distances",
    "CuspSlopeDiagnostic",
    "DiagnosticContext",
    "DiagnosticResult",
    "ExchangeSymmetryDiagnostic",
    "HistogramDiagnostic",
    "PairDistanceHistogramDiagnostic",
    "ParticleAntisymmetryDiagnostic",
    "RadialCutDiagnostic",
    "RadialDensityDiagnostic",
    "RadialLogAbsComparison",
    "SpinResolvedCuspSlopeDiagnostic",
]
