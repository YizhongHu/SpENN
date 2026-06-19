"""Typed primitive outputs shared across evaluation components.

``EvaluationBundle`` is intentionally not a generic dict.

A field belongs here only if it is a reusable primitive output produced by
calculators and consumed by multiple summaries/tasks. Derived diagnostic metrics
do not belong in the bundle.

Good bundle fields include:

- generated configurations
- wavefunction values
- local-energy values
- derivative values
- transform comparison values
- trace records

Do not add fields such as ``cusp_even_slope``, ``c_minus_1_abs``,
``tail_outlier_count``, ``pfaffian_near_zero_count``, or ``feature_rms_q95``.
Those are summary outputs, not shared primitive calculator outputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from spenn.data.batch import ElectronBatch
from spenn.trace import Trace


@dataclass(frozen=True)
class GeneratedConfigurations:
    """Electron configurations produced by an evaluation generator.

    Metadata is bookkeeping only. Scientific quantities computed from a model
    or Hamiltonian belong in calculator outputs, not in this object.
    """

    batch: ElectronBatch
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class WavefunctionValues:
    """Wavefunction values evaluated on generated configurations."""

    logabs: torch.Tensor
    sign: torch.Tensor
    components: Mapping[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class LocalEnergyValues:
    """Local-energy values evaluated on generated configurations."""

    local_energy: torch.Tensor
    finite_mask: torch.Tensor
    term_energies: Mapping[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class DerivativeValues:
    """Radial derivative values used by geometry summaries."""

    radial_dlogabs: torch.Tensor
    r12: torch.Tensor
    direction_id: torch.Tensor
    antipodal_pair_id: torch.Tensor | None = None
    direction_sign: torch.Tensor | None = None


@dataclass(frozen=True)
class TransformComparisonValues:
    """Raw values comparing original and transformed model outputs."""

    original_logabs: torch.Tensor
    transformed_logabs: torch.Tensor
    original_sign: torch.Tensor
    transformed_sign: torch.Tensor
    logabs_abs_error: torch.Tensor
    sign_mismatch: torch.Tensor
    metadata: Mapping[str, Any]
    local_energy_abs_error: torch.Tensor | None = None


@dataclass(frozen=True)
class TraceComparisonValues:
    """Raw trace-comparison records produced by trace equivariance checks."""

    max_abs_error: torch.Tensor
    mean_abs_error: torch.Tensor
    failure_count: int
    compared_entry_count: int
    comparison_error_count: int
    missing_key_count: int
    extra_key_count: int
    records: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class FeatureTraceValues:
    """Raw feature magnitude records collected from trace entries."""

    records: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class ReadoutTraceValues:
    """Raw readout conditioning records collected from trace entries."""

    records: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class EvaluationBundle:
    """Reusable primitive outputs for one evaluation task."""

    generated: GeneratedConfigurations
    wavefunction: WavefunctionValues | None = None
    local_energy: LocalEnergyValues | None = None
    derivatives: Mapping[str, DerivativeValues] | None = None
    trace: Trace | None = None
    transform: TransformComparisonValues | None = None
    trace_comparison: TraceComparisonValues | None = None
    feature_trace: FeatureTraceValues | None = None
    readout_trace: ReadoutTraceValues | None = None


__all__ = [
    "DerivativeValues",
    "EvaluationBundle",
    "FeatureTraceValues",
    "GeneratedConfigurations",
    "LocalEnergyValues",
    "ReadoutTraceValues",
    "TraceComparisonValues",
    "TransformComparisonValues",
    "WavefunctionValues",
]
