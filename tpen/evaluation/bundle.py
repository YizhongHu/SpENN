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
- electron-nucleus radial values
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

from tpen.data.batch import ElectronBatch
from tpen.data.batch.geometry import electron_nuclear_displacements
from tpen.data.equivariant_state import compare_tensor_blocks
from tpen.data.indices import permute_particle_axis
from tpen.data.permutation import Permutation
from tpen.trace import Trace


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
class ElectronNucleusRadialValues:
    """Typed electron-nucleus radial log-amplitude derivatives.

    Every field has shape ``[batch, n_electrons, n_nuclei]``. Distances use
    the raw geometry and must therefore be strictly positive for this
    diagnostic. ``finite_mask`` records derivative availability explicitly;
    nonfinite derivative values are permitted only when marked unavailable.
    """

    distance: torch.Tensor
    radial_dlogabs: torch.Tensor
    finite_mask: torch.Tensor

    def validate(self, batch: ElectronBatch) -> "ElectronNucleusRadialValues":
        """Validate shape, geometry, device, dtype, and availability semantics."""

        flat = batch.flatten_samples()
        if flat.nuclear_positions is None or flat.nuclear_charges is None:
            raise ValueError("ElectronNucleusRadialValues requires batch nuclear positions and charges")
        expected_shape = (flat.batch_size, flat.n_electrons, flat.nuclear_positions.shape[-2])
        for name, value in (("distance", self.distance), ("radial_dlogabs", self.radial_dlogabs)):
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"ElectronNucleusRadialValues.{name} must have shape {expected_shape}")
            if value.device != flat.device or value.dtype != flat.dtype:
                raise ValueError(
                    f"ElectronNucleusRadialValues.{name} must match batch dtype/device"
                )
        if tuple(self.finite_mask.shape) != expected_shape or self.finite_mask.dtype != torch.bool:
            raise ValueError(
                "ElectronNucleusRadialValues.finite_mask must be bool with shape "
                f"{expected_shape}"
            )
        if self.finite_mask.device != flat.device:
            raise ValueError("ElectronNucleusRadialValues.finite_mask must match batch device")
        if not torch.isfinite(self.distance).all() or torch.any(self.distance <= 0):
            raise ValueError("ElectronNucleusRadialValues.distance must be finite and strictly positive")
        expected_distance = electron_nuclear_displacements(flat).norm(dim=-1)
        if not torch.equal(self.distance, expected_distance):
            raise ValueError("ElectronNucleusRadialValues.distance must equal raw batch geometry")
        if not torch.equal(self.finite_mask, torch.isfinite(self.radial_dlogabs)):
            raise ValueError(
                "ElectronNucleusRadialValues.finite_mask must equal isfinite(radial_dlogabs)"
            )
        return self

    def permute(self, permutation: Permutation) -> "ElectronNucleusRadialValues":
        """Apply an electron permutation to every electron-indexed field."""

        if len(permutation) != self.distance.shape[1]:
            raise ValueError(
                f"Permutation of size {len(permutation)} is incompatible with "
                f"{self.distance.shape[1]} electrons"
            )
        return type(self)(
            distance=permute_particle_axis(self.distance, permutation, axis=1),
            radial_dlogabs=permute_particle_axis(self.radial_dlogabs, permutation, axis=1),
            finite_mask=permute_particle_axis(self.finite_mask, permutation, axis=1),
        )

    def compare(
        self,
        other: "ElectronNucleusRadialValues",
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, int | float]]:
        """Compare raw distances, availability masks, and finite derivatives."""

        if type(self) is not type(other):
            return False, {"max_abs_error": float("inf"), "finite_mask_mismatch_count": 0}
        self_shapes = (self.distance.shape, self.radial_dlogabs.shape, self.finite_mask.shape)
        other_shapes = (other.distance.shape, other.radial_dlogabs.shape, other.finite_mask.shape)
        self_devices = (self.distance.device, self.radial_dlogabs.device, self.finite_mask.device)
        other_devices = (other.distance.device, other.radial_dlogabs.device, other.finite_mask.device)
        self_dtypes = (self.distance.dtype, self.radial_dlogabs.dtype, self.finite_mask.dtype)
        other_dtypes = (other.distance.dtype, other.radial_dlogabs.dtype, other.finite_mask.dtype)
        if self_shapes != other_shapes or self_devices != other_devices or self_dtypes != other_dtypes:
            return False, {"max_abs_error": float("inf"), "finite_mask_mismatch_count": 0}
        mask_mismatch_count = int((self.finite_mask != other.finite_mask).sum().item())
        distance_close, distance_metrics = compare_tensor_blocks(
            [self.distance],
            [other.distance],
            atol=atol,
            rtol=rtol,
        )
        shared_finite = self.finite_mask & other.finite_mask
        derivative_close, derivative_metrics = compare_tensor_blocks(
            [self.radial_dlogabs[shared_finite]],
            [other.radial_dlogabs[shared_finite]],
            atol=atol,
            rtol=rtol,
        )
        max_abs_error = max(
            float(distance_metrics["max_abs_error"]),
            float(derivative_metrics["max_abs_error"]),
        )
        return (
            distance_close and derivative_close and mask_mismatch_count == 0,
            {
                "max_abs_error": max_abs_error,
                "finite_mask_mismatch_count": mask_mismatch_count,
            },
        )


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
    electron_nucleus_radial: ElectronNucleusRadialValues | None = None
    trace: Trace | None = None
    transform: TransformComparisonValues | None = None
    trace_comparison: TraceComparisonValues | None = None
    feature_trace: FeatureTraceValues | None = None
    readout_trace: ReadoutTraceValues | None = None


__all__ = [
    "DerivativeValues",
    "ElectronNucleusRadialValues",
    "EvaluationBundle",
    "FeatureTraceValues",
    "GeneratedConfigurations",
    "LocalEnergyValues",
    "ReadoutTraceValues",
    "TraceComparisonValues",
    "TransformComparisonValues",
    "WavefunctionValues",
]
