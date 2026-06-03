"""Reusable metric and plot-data diagnostics."""

from spenn.diagnostics.base import DiagnosticContext, DiagnosticResult
from spenn.diagnostics.statistics import (
    autocorrelation_by_lag,
    effective_sample_size,
    integrated_autocorrelation_time,
)
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
    "autocorrelation_by_lag",
    "CuspSlopeDiagnostic",
    "DiagnosticContext",
    "DiagnosticResult",
    "effective_sample_size",
    "ExchangeSymmetryDiagnostic",
    "HistogramDiagnostic",
    "integrated_autocorrelation_time",
    "PairDistanceHistogramDiagnostic",
    "ParticleAntisymmetryDiagnostic",
    "RadialCutDiagnostic",
    "RadialDensityDiagnostic",
    "RadialLogAbsComparison",
    "SpinResolvedCuspSlopeDiagnostic",
]
