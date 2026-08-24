"""Summaries and full-row artifacts for typed helium atlas primitives."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence

import torch

from tpen.evaluation.bundle import EvaluationBundle, HeliumAtlasValues
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.results import ArtifactRecord, MetricScalar, SummaryResult


class HeliumCurvatureSummary:
    """Summarize direct second derivatives in predeclared nested windows.

    No universal Kato curvature target is defined or fitted. The windows are
    named by configuration and validated as strictly nested before evaluation.
    """

    name = "helium_curvature"
    required_fields = frozenset({"helium_atlas"})

    def __init__(
        self,
        *,
        series_name: str,
        windows: Mapping[str, float],
        metric_prefix: str,
    ) -> None:
        self.series_name = _validated_label(series_name, name="series_name")
        self.metric_prefix = _validated_label(metric_prefix, name="metric_prefix")
        declared: list[tuple[str, float]] = []
        for name, upper_bound in windows.items():
            label = _validated_label(name, name="curvature window name")
            bound = float(upper_bound)
            if not math.isfinite(bound) or bound <= 0:
                raise ValueError("HeliumCurvatureSummary window bounds must be finite and positive")
            declared.append((label, bound))
        if len(declared) < 2:
            raise ValueError("HeliumCurvatureSummary requires at least two predeclared nested windows")
        if any(right[1] <= left[1] for left, right in zip(declared, declared[1:])):
            raise ValueError(
                "HeliumCurvatureSummary windows must be predeclared in strictly nested increasing order"
            )
        self.windows = tuple(declared)

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return finite-aware direct-curvature values for every named window."""

        del context, namespace
        atlas = _atlas(bundle)
        values = atlas.derivatives.get(self.series_name)
        if values is None:
            raise ValueError(f"helium atlas has no derivative series {self.series_name!r}")
        direction = _long_metadata(bundle, "direction_id")
        eligible = (~atlas.is_exact_zero_sentinel) & (atlas.realized_coordinate > 0)
        metrics: dict[str, MetricScalar] = {}
        for window_name, upper_bound in self.windows:
            in_window = eligible & (atlas.realized_coordinate <= upper_bound)
            finite = in_window & values.second_derivative_finite_mask
            prefix = f"{self.metric_prefix}_{window_name}"
            metrics[f"{prefix}_total_count"] = int(in_window.sum().item())
            metrics[f"{prefix}_finite_count"] = int(finite.sum().item())
            metrics[f"{prefix}_nonfinite_count"] = int((in_window & ~finite).sum().item())
            metrics[f"{prefix}_available"] = bool(finite.any().item())
            if finite.any():
                curvature = values.second_derivative[finite]
                directional_means = _directional_means(
                    values.second_derivative, direction=direction, selection=finite
                )
                metrics[f"{prefix}_second_derivative_mean"] = float(curvature.mean().item())
                metrics[f"{prefix}_second_derivative_min"] = float(curvature.min().item())
                metrics[f"{prefix}_second_derivative_max"] = float(curvature.max().item())
                metrics[f"{prefix}_directional_spread"] = float(
                    (directional_means.max() - directional_means.min()).item()
                )
        return SummaryResult(metrics=metrics)


class HeliumTailSummary:
    """Emit the five required named outer-tail quantities."""

    name = "helium_tail"
    required_fields = frozenset({"helium_atlas"})

    def __init__(self, *, series_name: str, metric_prefix: str) -> None:
        self.series_name = _validated_label(series_name, name="series_name")
        self.metric_prefix = _validated_label(metric_prefix, name="metric_prefix")

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return slope, extrema, sign fraction, outer radius, and spread."""

        del context, namespace
        atlas = _atlas(bundle)
        values = atlas.derivatives.get(self.series_name)
        if values is None:
            raise ValueError(f"helium atlas has no derivative series {self.series_name!r}")
        direction = _long_metadata(bundle, "direction_id")
        considered = (~atlas.is_exact_zero_sentinel) & torch.isfinite(
            atlas.realized_coordinate
        )
        eligible = considered & values.first_derivative_finite_mask
        outer_slopes: list[torch.Tensor] = []
        outer_radii: list[torch.Tensor] = []
        for direction_id in sorted(set(direction.tolist())):
            selected = eligible & (direction == direction_id)
            if not selected.any():
                continue
            group_indices = torch.nonzero(selected, as_tuple=False).reshape(-1)
            outer_radius = atlas.realized_coordinate[group_indices].max()
            outer_indices = group_indices[
                atlas.realized_coordinate[group_indices] == outer_radius
            ]
            outer_slopes.append(values.first_derivative[outer_indices].mean())
            outer_radii.append(outer_radius)
        metrics: dict[str, MetricScalar] = {
            f"{self.metric_prefix}_total_count": int(considered.sum().item()),
            f"{self.metric_prefix}_finite_count": int(eligible.sum().item()),
            f"{self.metric_prefix}_nonfinite_count": int(
                (considered & ~values.first_derivative_finite_mask).sum().item()
            ),
            f"{self.metric_prefix}_direction_count": len(outer_slopes),
            f"{self.metric_prefix}_available": bool(outer_slopes),
        }
        if outer_slopes:
            slopes = torch.stack(outer_slopes)
            radii = torch.stack(outer_radii)
            # Five named contract quantities. Extrema is represented by its
            # explicit lower and upper endpoints rather than an opaque tuple.
            metrics[f"{self.metric_prefix}_slope"] = float(slopes.mean().item())
            metrics[f"{self.metric_prefix}_extrema_min"] = float(slopes.min().item())
            metrics[f"{self.metric_prefix}_extrema_max"] = float(slopes.max().item())
            metrics[f"{self.metric_prefix}_sign_fraction"] = float(
                (slopes < 0).to(dtype=slopes.dtype).mean().item()
            )
            metrics[f"{self.metric_prefix}_outer_radius"] = float(radii.max().item())
            metrics[f"{self.metric_prefix}_directional_spread"] = float(
                (slopes.max() - slopes.min()).item()
            )
        return SummaryResult(metrics=metrics)


class HeliumNumericalLimitSummary:
    """Count every retained numerical status without dropping nonfinite rows."""

    name = "helium_numerical_limits"
    required_fields = frozenset({"helium_atlas"})

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return sentinel, boundary, domain, and finite-status counts."""

        del context, namespace
        atlas = _atlas(bundle)
        sample_nonfinite = ~atlas.total_local_energy_finite_mask
        for values in atlas.derivatives.values():
            sample_nonfinite |= ~values.value_finite_mask
            sample_nonfinite |= ~values.first_derivative_finite_mask
            sample_nonfinite |= ~values.second_derivative_finite_mask
        for mask in atlas.hamiltonian_term_finite_masks.values():
            sample_nonfinite |= ~mask
        sample_nonfinite |= ~atlas.per_electron_kinetic_domain_mask.all(dim=1)
        sample_nonfinite |= ~atlas.cancellation_abs_sum_finite_mask
        sample_nonfinite |= ~atlas.cancellation_residual_finite_mask
        sample_nonfinite |= ~atlas.cancellation_ratio_finite_mask
        metrics: dict[str, MetricScalar] = {
            "atlas_total_count": int(atlas.requested_coordinate.numel()),
            "atlas_refinement_boundary_count": int(atlas.is_refinement_boundary.sum().item()),
            "atlas_exact_zero_sentinel_count": int(atlas.is_exact_zero_sentinel.sum().item()),
            "atlas_computed_nonfinite_retained_count": int(sample_nonfinite.sum().item()),
            "atlas_dtype_is_float64": atlas.provenance.dtype == "float64",
            "atlas_seed": int(atlas.provenance.seed),
        }
        if atlas.ideal_unfloored_ee_inverse_distance is not None:
            ideal = atlas.ideal_unfloored_ee_inverse_distance
            ideal_domain = atlas.ideal_unfloored_ee_domain_mask
            assert ideal_domain is not None
            executed = atlas.derivatives.get("executed_smoothed_ee_factor")
            if executed is None:
                raise ValueError(
                    "an e-e numerical atlas requires executed_smoothed_ee_factor diagnostics"
                )
            ideal_finite = torch.isfinite(ideal)
            metrics.update(
                {
                    "ideal_unfloored_ee_domain_count": int(ideal_domain.sum().item()),
                    "ideal_unfloored_ee_nonfinite_count": int((~ideal_finite).sum().item()),
                    "executed_smoothed_ee_factor_finite_at_ideal_undefined_count": int(
                        (executed.value_finite_mask & ~ideal_domain).sum().item()
                    ),
                }
            )
        return SummaryResult(metrics=metrics)


class HeliumAtlasWriter:
    """Write every atlas row, including nonfinite and exact-zero sentinels."""

    name = "helium_atlas_records"
    required_fields = frozenset({"helium_atlas"})

    def __init__(self, *, enabled: bool = True, filename: str = "helium_atlas.csv") -> None:
        self.enabled = bool(enabled)
        self.filename = str(filename)

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Write a lossless finite/domain-status atlas table."""

        del namespace
        atlas = _atlas(bundle)
        if not self.enabled or context.artifact_level == "metrics_only":
            return SummaryResult(metrics={})
        metadata = bundle.generated.metadata
        coordinate_kinds = _string_metadata(bundle, "atlas_coordinate_kind")
        geometry_kinds = _string_metadata(bundle, "atlas_geometry_kind")
        sample_kinds = _string_metadata(bundle, "atlas_sample_kind")
        direction = _long_metadata(bundle, "direction_id")
        ray = _long_metadata(bundle, "ray_id")
        refinement = _long_metadata(bundle, "refinement_index")
        probe = _long_metadata(bundle, "probe_electron")
        del metadata

        derivative_columns: list[str] = []
        for name in atlas.derivatives:
            derivative_columns.extend(
                [
                    f"{name}_value",
                    f"{name}_value_finite",
                    f"{name}_first_derivative",
                    f"{name}_first_derivative_finite",
                    f"{name}_second_derivative",
                    f"{name}_second_derivative_finite",
                ]
            )
        term_columns = [_term_column(name) for name in atlas.hamiltonian_terms]
        fields = [
            "sample_index",
            "atlas_coordinate_kind",
            "atlas_geometry_kind",
            "atlas_sample_kind",
            "direction_id",
            "ray_id",
            "refinement_index",
            "probe_electron",
            "boundary_provenance_dtype",
            "boundary_provenance_device",
            "boundary_provenance_seed",
            "evaluation_dtype",
            "evaluation_device",
            "requested_path_coordinate",
            "realized_physical_coordinate",
            "is_refinement_boundary",
            "is_exact_zero_sentinel",
            "domain_status",
            "ideal_unfloored_ee_inverse_distance",
            "ideal_unfloored_ee_domain",
            "ideal_unfloored_ee_finite",
            *derivative_columns,
            "executed_hamiltonian_total",
            "executed_hamiltonian_total_finite",
            *term_columns,
            *[f"{column}_finite" for column in term_columns],
            "executed_per_electron_kinetic_0",
            "executed_per_electron_kinetic_0_status",
            "executed_per_electron_kinetic_1",
            "executed_per_electron_kinetic_1_status",
            "executed_hamiltonian_cancellation_abs_sum",
            "executed_hamiltonian_cancellation_abs_sum_finite",
            "executed_hamiltonian_cancellation_residual",
            "executed_hamiltonian_cancellation_residual_finite",
            "executed_hamiltonian_cancellation_ratio",
            "executed_hamiltonian_cancellation_ratio_finite",
        ]
        path = context.task_output_dir / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for index in range(atlas.requested_coordinate.numel()):
                ideal = atlas.ideal_unfloored_ee_inverse_distance
                ideal_domain = atlas.ideal_unfloored_ee_domain_mask
                row: dict[str, object] = {
                    "sample_index": index,
                    "atlas_coordinate_kind": coordinate_kinds[index],
                    "atlas_geometry_kind": geometry_kinds[index],
                    "atlas_sample_kind": sample_kinds[index],
                    "direction_id": int(direction[index].item()),
                    "ray_id": int(ray[index].item()),
                    "refinement_index": int(refinement[index].item()),
                    "probe_electron": int(probe[index].item()),
                    "boundary_provenance_dtype": atlas.provenance.dtype,
                    "boundary_provenance_device": atlas.provenance.device,
                    "boundary_provenance_seed": atlas.provenance.seed,
                    "evaluation_dtype": atlas.provenance.evaluation_dtype,
                    "evaluation_device": atlas.provenance.evaluation_device,
                    "requested_path_coordinate": _number(atlas.requested_coordinate[index]),
                    "realized_physical_coordinate": _number(atlas.realized_coordinate[index]),
                    "is_refinement_boundary": bool(atlas.is_refinement_boundary[index].item()),
                    "is_exact_zero_sentinel": bool(atlas.is_exact_zero_sentinel[index].item()),
                    "domain_status": atlas.domain_status[index],
                    "ideal_unfloored_ee_inverse_distance": (
                        "not_applicable" if ideal is None else _number(ideal[index])
                    ),
                    "ideal_unfloored_ee_domain": (
                        "not_applicable"
                        if ideal_domain is None
                        else bool(ideal_domain[index].item())
                    ),
                    "ideal_unfloored_ee_finite": (
                        "not_applicable"
                        if ideal is None
                        else bool(torch.isfinite(ideal[index]).item())
                    ),
                    "executed_hamiltonian_total": _number(atlas.total_local_energy[index]),
                    "executed_hamiltonian_total_finite": bool(
                        atlas.total_local_energy_finite_mask[index].item()
                    ),
                    "executed_per_electron_kinetic_0": _number(
                        atlas.per_electron_kinetic[index, 0]
                    ),
                    "executed_per_electron_kinetic_0_status": atlas.per_electron_kinetic_status[index][0],
                    "executed_per_electron_kinetic_1": _number(
                        atlas.per_electron_kinetic[index, 1]
                    ),
                    "executed_per_electron_kinetic_1_status": atlas.per_electron_kinetic_status[index][1],
                    "executed_hamiltonian_cancellation_abs_sum": _number(
                        atlas.cancellation_abs_sum[index]
                    ),
                    "executed_hamiltonian_cancellation_abs_sum_finite": bool(
                        atlas.cancellation_abs_sum_finite_mask[index].item()
                    ),
                    "executed_hamiltonian_cancellation_residual": _number(
                        atlas.cancellation_residual[index]
                    ),
                    "executed_hamiltonian_cancellation_residual_finite": bool(
                        atlas.cancellation_residual_finite_mask[index].item()
                    ),
                    "executed_hamiltonian_cancellation_ratio": _number(
                        atlas.cancellation_ratio[index]
                    ),
                    "executed_hamiltonian_cancellation_ratio_finite": bool(
                        atlas.cancellation_ratio_finite_mask[index].item()
                    ),
                }
                for name, values in atlas.derivatives.items():
                    row[f"{name}_value"] = _number(values.value[index])
                    row[f"{name}_value_finite"] = bool(values.value_finite_mask[index].item())
                    row[f"{name}_first_derivative"] = _number(values.first_derivative[index])
                    row[f"{name}_first_derivative_finite"] = bool(
                        values.first_derivative_finite_mask[index].item()
                    )
                    row[f"{name}_second_derivative"] = _number(values.second_derivative[index])
                    row[f"{name}_second_derivative_finite"] = bool(
                        values.second_derivative_finite_mask[index].item()
                    )
                for name, values in atlas.hamiltonian_terms.items():
                    column = _term_column(name)
                    row[column] = _number(values[index])
                    row[f"{column}_finite"] = bool(
                        atlas.hamiltonian_term_finite_masks[name][index].item()
                    )
                writer.writerow(row)
        count = int(atlas.requested_coordinate.numel())
        return SummaryResult(
            metrics={"helium_atlas_row_count": count},
            artifacts=(
                ArtifactRecord(
                    name="helium_atlas",
                    kind="csv",
                    path=path,
                    metadata={
                        "rows": count,
                        "refinement_boundary_count": int(
                            atlas.is_refinement_boundary.sum().item()
                        ),
                        "exact_zero_sentinel_count": int(
                            atlas.is_exact_zero_sentinel.sum().item()
                        ),
                        "boundary_provenance_dtype": atlas.provenance.dtype,
                        "boundary_provenance_device": atlas.provenance.device,
                        "boundary_provenance_seed": atlas.provenance.seed,
                        "evaluation_dtype": atlas.provenance.evaluation_dtype,
                        "evaluation_device": atlas.provenance.evaluation_device,
                    },
                ),
            ),
        )


def _atlas(bundle: EvaluationBundle) -> HeliumAtlasValues:
    atlas = bundle.helium_atlas
    if not isinstance(atlas, HeliumAtlasValues):
        raise ValueError("helium atlas summaries require bundle.helium_atlas")
    return atlas.validate(bundle.generated.batch)


def _long_metadata(bundle: EvaluationBundle, key: str) -> torch.Tensor:
    batch = bundle.generated.batch.flatten_samples()
    value = bundle.generated.metadata.get(key)
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (batch.batch_size,)
        or value.dtype != torch.long
        or value.device != batch.device
    ):
        raise ValueError(f"{key} metadata must be long with shape ({batch.batch_size},)")
    return value


def _string_metadata(bundle: EvaluationBundle, key: str) -> tuple[str, ...]:
    batch_size = bundle.generated.batch.flatten_samples().batch_size
    value = bundle.generated.metadata.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} metadata must be a sequence")
    resolved = tuple(value)
    if len(resolved) != batch_size or any(not isinstance(item, str) for item in resolved):
        raise ValueError(f"{key} metadata must contain {batch_size} strings")
    return resolved


def _directional_means(
    values: torch.Tensor,
    *,
    direction: torch.Tensor,
    selection: torch.Tensor,
) -> torch.Tensor:
    means = [
        values[selection & (direction == direction_id)].mean()
        for direction_id in sorted(set(direction[selection].tolist()))
    ]
    return torch.stack(means)


def _validated_label(value: str, *, name: str) -> str:
    label = str(value)
    if not label or not label.strip():
        raise ValueError(f"Helium atlas {name} must be non-empty")
    return label


def _number(value: torch.Tensor) -> float | str:
    number = float(value.item())
    if math.isfinite(number):
        return number
    return "inf" if number > 0 else "-inf" if number < 0 else "nan"


def _term_column(name: str) -> str:
    if name == "electron_electron":
        return "executed_smoothed_physical_separation_hamiltonian_term/electron_electron"
    return f"executed_hamiltonian_term/{name}"


__all__ = [
    "HeliumAtlasWriter",
    "HeliumCurvatureSummary",
    "HeliumNumericalLimitSummary",
    "HeliumTailSummary",
]
