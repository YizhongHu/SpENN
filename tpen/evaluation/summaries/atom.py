"""Atom-owned radial summaries and profile artifacts."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tpen.evaluation.bundle import ElectronNucleusRadialValues, EvaluationBundle
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.results import ArtifactRecord, MetricScalar, SummaryResult


class ElectronNucleusCuspSummary:
    """Extrapolate antipodally averaged one-sided cusp slopes to ``r -> 0+``."""

    name = "electron_nucleus_cusp"
    required_fields = frozenset({"electron_nucleus_radial"})

    def __init__(self, *, max_fit_points: int | None = None) -> None:
        self.max_fit_points = None if max_fit_points is None else int(max_fit_points)
        if self.max_fit_points is not None and self.max_fit_points < 2:
            raise ValueError("ElectronNucleusCuspSummary max_fit_points must be at least 2")

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return availability, finite counts, and one-sided cusp estimates."""

        del context, namespace
        measurements = _profile_measurements(bundle, region="cusp")
        metrics = _availability_metrics("cusp", measurements)
        estimates: list[torch.Tensor] = []
        expected: list[torch.Tensor] = []
        for key in measurements.group_keys():
            mask = measurements.group_mask(key) & measurements.finite
            if int(mask.sum().item()) < 2:
                continue
            radius = measurements.radius[mask]
            slope = measurements.radial_dlogabs[mask]
            order = torch.argsort(radius)
            if self.max_fit_points is not None:
                order = order[: self.max_fit_points]
            radius = radius[order]
            slope = slope[order]
            if torch.unique(radius).numel() < 2:
                continue
            design = torch.stack((torch.ones_like(radius), radius), dim=-1)
            intercept = torch.linalg.lstsq(design, slope.unsqueeze(-1)).solution[0, 0]
            group_expected = measurements.expected_slope[mask][order]
            if not torch.allclose(group_expected, group_expected[:1].expand_as(group_expected)):
                raise ValueError("cusp fit group changed nuclear charge")
            if torch.isfinite(intercept):
                estimates.append(intercept)
                expected.append(group_expected[0])

        metrics["cusp_finite_fit_count"] = len(estimates)
        metrics["cusp_available"] = bool(estimates)
        if estimates:
            fitted = torch.stack(estimates)
            expected_slope = torch.stack(expected)
            error = (fitted - expected_slope).abs()
            metrics.update(
                {
                    "cusp_expected_slope": float(expected_slope.mean().item()),
                    "cusp_one_sided_slope_mean": float(fitted.mean().item()),
                    "cusp_one_sided_slope_abs_error_mean": float(error.mean().item()),
                    "cusp_one_sided_slope_abs_error_max": float(error.max().item()),
                }
            )
        return SummaryResult(metrics=metrics)


class ElectronNucleusTailSummary:
    """Summarize antipodally averaged outer electron-nucleus decay slopes."""

    name = "electron_nucleus_tail"
    required_fields = frozenset({"electron_nucleus_radial"})

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return availability, finite counts, and explicit outer-ray slopes."""

        del context, namespace
        measurements = _profile_measurements(bundle, region="tail")
        metrics = _availability_metrics("tail", measurements)
        outer_slopes: list[torch.Tensor] = []
        outer_radii: list[torch.Tensor] = []
        for key in measurements.group_keys():
            mask = measurements.group_mask(key) & measurements.finite
            if not torch.any(mask):
                continue
            group_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
            outer_radius = measurements.radius[group_indices].max()
            outer_indices = group_indices[measurements.radius[group_indices] == outer_radius]
            outer_slopes.append(measurements.radial_dlogabs[outer_indices].mean())
            outer_radii.append(outer_radius)

        metrics["tail_outer_measurement_count"] = len(outer_slopes)
        metrics["tail_available"] = bool(outer_slopes)
        if outer_slopes:
            slopes = torch.stack(outer_slopes)
            radii = torch.stack(outer_radii)
            metrics.update(
                {
                    "tail_outer_slope_mean": float(slopes.mean().item()),
                    "tail_outer_slope_min": float(slopes.min().item()),
                    "tail_outer_slope_max": float(slopes.max().item()),
                    "tail_outer_radius_min": float(radii.min().item()),
                    "tail_outer_radius_max": float(radii.max().item()),
                    "tail_negative_slope_fraction": float((slopes < 0).to(slopes.dtype).mean().item()),
                }
            )
        return SummaryResult(metrics=metrics)


class ElectronNucleusRadialProfileWriter:
    """Write the designated He electron-nucleus derivative profile as CSV."""

    name = "electron_nucleus_radial_profile"
    required_fields = frozenset({"electron_nucleus_radial"})

    def __init__(self, *, enabled: bool = True, filename: str = "electron_nucleus_radial_profile.csv") -> None:
        self.enabled = bool(enabled)
        self.filename = str(filename)

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Write finite and unavailable profile rows with explicit counts."""

        del namespace
        measurements = _profile_measurements(bundle, region=None)
        metrics = _availability_metrics("profile", measurements)
        if not self.enabled or context.artifact_level == "metrics_only":
            return SummaryResult(metrics=metrics)

        path = context.task_output_dir / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "measurement_index",
                    "profile_region",
                    "direction_id",
                    "direction_sign",
                    "probe_electron",
                    "nucleus_index",
                    "radius",
                    "radial_dlogabs",
                    "finite",
                    "available",
                    "finite_measurement_count",
                    "total_measurement_count",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            for index in range(measurements.count):
                finite = bool(measurements.finite[index].item())
                writer.writerow(
                    {
                        "measurement_index": int(measurements.sample_index[index].item()),
                        "profile_region": measurements.region[index],
                        "direction_id": int(measurements.direction_id[index].item()),
                        "direction_sign": int(measurements.direction_sign[index].item()),
                        "probe_electron": int(measurements.probe_electron[index].item()),
                        "nucleus_index": int(measurements.nucleus_index[index].item()),
                        "radius": float(measurements.radius[index].item()),
                        "radial_dlogabs": (
                            float(measurements.radial_dlogabs[index].item()) if finite else "unavailable"
                        ),
                        "finite": finite,
                        "available": finite,
                        "finite_measurement_count": int(metrics["profile_finite_measurement_count"]),
                        "total_measurement_count": int(metrics["profile_total_measurement_count"]),
                    }
                )
        return SummaryResult(
            metrics=metrics,
            artifacts=(
                ArtifactRecord(
                    name="electron_nucleus_radial_profile",
                    kind="csv",
                    path=path,
                    metadata={
                        "available": bool(metrics["profile_available"]),
                        "finite_measurement_count": int(metrics["profile_finite_measurement_count"]),
                        "total_measurement_count": int(metrics["profile_total_measurement_count"]),
                    },
                ),
            ),
        )


@dataclass(frozen=True)
class _ProfileMeasurements:
    sample_index: torch.Tensor
    region: tuple[str, ...]
    radius: torch.Tensor
    radial_dlogabs: torch.Tensor
    finite: torch.Tensor
    direction_id: torch.Tensor
    direction_sign: torch.Tensor
    probe_electron: torch.Tensor
    nucleus_index: torch.Tensor
    expected_slope: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.radius.numel())

    def group_keys(self) -> set[tuple[int, int, int]]:
        return {
            (int(direction), int(electron), int(nucleus))
            for direction, electron, nucleus in zip(
                self.direction_id.tolist(),
                self.probe_electron.tolist(),
                self.nucleus_index.tolist(),
                strict=True,
            )
        }

    def group_mask(self, key: tuple[int, int, int]) -> torch.Tensor:
        direction, electron, nucleus = key
        return (
            (self.direction_id == direction)
            & (self.probe_electron == electron)
            & (self.nucleus_index == nucleus)
        )


def _profile_measurements(
    bundle: EvaluationBundle,
    *,
    region: str | None,
) -> _ProfileMeasurements:
    values = bundle.electron_nucleus_radial
    if not isinstance(values, ElectronNucleusRadialValues):
        raise ValueError("atom radial summaries require bundle.electron_nucleus_radial")
    flat = bundle.generated.batch.flatten_samples()
    values.validate(flat)
    metadata = bundle.generated.metadata
    regions_raw = metadata.get("profile_region")
    if not isinstance(regions_raw, Sequence) or isinstance(regions_raw, (str, bytes)):
        raise ValueError("profile_region metadata must be a sequence")
    regions = tuple(str(value) for value in regions_raw)
    if len(regions) != flat.batch_size:
        raise ValueError("profile_region metadata must match generated batch size")
    direction_id = _long_metadata(metadata, "direction_id", batch_size=flat.batch_size, like=values.distance)
    direction_sign = _long_metadata(metadata, "direction_sign", batch_size=flat.batch_size, like=values.distance)
    probe_electron = _long_metadata(metadata, "probe_electron", batch_size=flat.batch_size, like=values.distance)
    nucleus_index = _long_metadata(metadata, "nucleus_index", batch_size=flat.batch_size, like=values.distance)
    if torch.any((probe_electron < 0) | (probe_electron >= flat.n_electrons)):
        raise ValueError("probe_electron metadata is out of range")
    if not torch.all((direction_sign == -1) | (direction_sign == 1)):
        raise ValueError("direction_sign metadata must contain only -1 or 1")
    if torch.any((nucleus_index < 0) | (nucleus_index >= values.distance.shape[-1])):
        raise ValueError("nucleus_index metadata is out of range")

    sample_index = torch.arange(flat.batch_size, device=flat.device)
    selected_distance = values.distance[sample_index, probe_electron, nucleus_index]
    selected_derivative = values.radial_dlogabs[sample_index, probe_electron, nucleus_index]
    selected_finite = values.finite_mask[sample_index, probe_electron, nucleus_index]
    charges = flat.nuclear_charges
    assert charges is not None
    expected_slope = -(
        charges[nucleus_index]
        if charges.ndim == 1
        else charges[sample_index, nucleus_index]
    )
    if region is None:
        selection = torch.ones(flat.batch_size, device=flat.device, dtype=torch.bool)
    else:
        selection = torch.tensor(
            [value == region for value in regions],
            device=flat.device,
            dtype=torch.bool,
        )
    indices = torch.nonzero(selection, as_tuple=False).reshape(-1)
    return _ProfileMeasurements(
        sample_index=sample_index[indices],
        region=tuple(regions[index] for index in indices.tolist()),
        radius=selected_distance[indices],
        radial_dlogabs=selected_derivative[indices],
        finite=selected_finite[indices],
        direction_id=direction_id[indices],
        direction_sign=direction_sign[indices],
        probe_electron=probe_electron[indices],
        nucleus_index=nucleus_index[indices],
        expected_slope=expected_slope[indices],
    )


def _long_metadata(
    metadata,
    key: str,
    *,
    batch_size: int,
    like: torch.Tensor,
) -> torch.Tensor:
    value = metadata.get(key)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch_size,):
        raise ValueError(f"{key} metadata must be a tensor with shape ({batch_size},)")
    return value.to(device=like.device, dtype=torch.long)


def _availability_metrics(
    prefix: str,
    measurements: _ProfileMeasurements,
) -> dict[str, MetricScalar]:
    finite_count = int(measurements.finite.sum().item())
    return {
        f"{prefix}_available": finite_count > 0,
        f"{prefix}_finite_measurement_count": finite_count,
        f"{prefix}_total_measurement_count": measurements.count,
    }


__all__ = [
    "ElectronNucleusCuspSummary",
    "ElectronNucleusRadialProfileWriter",
    "ElectronNucleusTailSummary",
]
