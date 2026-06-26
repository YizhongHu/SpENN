"""Hooke-pair deterministic evaluation diagnostics."""

from __future__ import annotations

import csv
import importlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from spenn.data.batch import ElectronBatch
from spenn.diagnostics.artifacts import update_diagnostic_index
from spenn.diagnostics.base import EvaluationContext, JsonScalar, evaluate_local_energy_in_chunks
from spenn.physics.hamiltonian import LocalEnergyResult
from spenn.physics.hooke import HookeSingletExact


class HookePairDistanceProbe:
    """Write a deterministic pair-distance probe for the Hooke singlet."""

    def __init__(
        self,
        name: str = "pair_distance_probe",
        artifact_path: str | Path | None = None,
        reference_energy: float | None = None,
        include_exact_wavefunction: bool = True,
        exact_wavefunction: object | str | None = None,
        r12_min: float = 1.0e-4,
        r12_max: float = 8.0,
        n_points: int = 200,
        n_directions: int = 3,
        center_of_mass_radii: Sequence[float] = (0.0, 0.5, 1.0),
        local_energy_chunk_size: int = 256,
        enabled: bool = True,
    ) -> None:
        self.name = str(name)
        self.artifact_path = None if artifact_path is None else Path(artifact_path)
        self.reference_energy = None if reference_energy is None else float(reference_energy)
        self.include_exact_wavefunction = bool(include_exact_wavefunction)
        self.exact_wavefunction = _resolve_exact_wavefunction(exact_wavefunction)
        self.r12_min = float(r12_min)
        self.r12_max = float(r12_max)
        self.n_points = int(n_points)
        self.n_directions = int(n_directions)
        self.center_of_mass_radii = tuple(float(value) for value in center_of_mass_radii)
        self.local_energy_chunk_size = int(local_energy_chunk_size)
        self.enabled = bool(enabled)

    def evaluate(self, context: EvaluationContext) -> Mapping[str, JsonScalar]:
        """Run the probe and return scalar summaries."""

        path = _artifact_path(context, self.artifact_path, self.name)
        if not self.enabled:
            _register_disabled(context, name=self.name, path=path, created_by=type(self).__name__)
            return {}
        positions = _pair_distance_positions(
            context,
            r12_min=self.r12_min,
            r12_max=self.r12_max,
            n_points=self.n_points,
            n_directions=self.n_directions,
            center_of_mass_radii=self.center_of_mass_radii,
        )
        rows = _evaluate_probe_rows(
            context,
            positions=positions,
            varying_key="pair_distance",
            fixed_key="center_of_mass_radius",
            reference_energy=self.reference_energy,
            include_exact_wavefunction=self.include_exact_wavefunction,
            exact_wavefunction=self.exact_wavefunction,
            local_energy_chunk_size=self.local_energy_chunk_size,
        )
        _write_probe_csv(path, rows, include_exact=self.include_exact_wavefunction)
        _register_artifact(context, name=self.name, path=path, created_by=type(self).__name__)
        summary = _probe_error_summary(rows, reference_energy=self.reference_energy)
        return {
            "probe_pair_distance/local_energy_max_abs_error": summary["max_abs_error"],
            "probe_pair_distance/local_energy_q95_abs_error": summary["q95_abs_error"],
            "probe_pair_distance/nonfinite_count": summary["nonfinite_count"],
        }


class HookePairCenterOfMassProbe:
    """Write a deterministic center-of-mass probe for the Hooke singlet."""

    def __init__(
        self,
        name: str = "center_of_mass_probe",
        artifact_path: str | Path | None = None,
        reference_energy: float | None = None,
        include_exact_wavefunction: bool = True,
        exact_wavefunction: object | str | None = None,
        com_radius_min: float = 0.0,
        com_radius_max: float = 8.0,
        n_points: int = 200,
        pair_distance: float = 1.0,
        n_directions: int = 3,
        local_energy_chunk_size: int = 256,
        enabled: bool = True,
    ) -> None:
        self.name = str(name)
        self.artifact_path = None if artifact_path is None else Path(artifact_path)
        self.reference_energy = None if reference_energy is None else float(reference_energy)
        self.include_exact_wavefunction = bool(include_exact_wavefunction)
        self.exact_wavefunction = _resolve_exact_wavefunction(exact_wavefunction)
        self.com_radius_min = float(com_radius_min)
        self.com_radius_max = float(com_radius_max)
        self.n_points = int(n_points)
        self.pair_distance = float(pair_distance)
        self.n_directions = int(n_directions)
        self.local_energy_chunk_size = int(local_energy_chunk_size)
        self.enabled = bool(enabled)

    def evaluate(self, context: EvaluationContext) -> Mapping[str, JsonScalar]:
        """Run the probe and return scalar summaries."""

        path = _artifact_path(context, self.artifact_path, self.name)
        if not self.enabled:
            _register_disabled(context, name=self.name, path=path, created_by=type(self).__name__)
            return {}
        positions = _center_of_mass_positions(
            context,
            com_radius_min=self.com_radius_min,
            com_radius_max=self.com_radius_max,
            n_points=self.n_points,
            pair_distance=self.pair_distance,
            n_directions=self.n_directions,
        )
        rows = _evaluate_probe_rows(
            context,
            positions=positions,
            varying_key="center_of_mass_radius",
            fixed_key="pair_distance",
            reference_energy=self.reference_energy,
            include_exact_wavefunction=self.include_exact_wavefunction,
            exact_wavefunction=self.exact_wavefunction,
            local_energy_chunk_size=self.local_energy_chunk_size,
        )
        _write_probe_csv(path, rows, include_exact=self.include_exact_wavefunction)
        _register_artifact(context, name=self.name, path=path, created_by=type(self).__name__)
        summary = _probe_error_summary(rows, reference_energy=self.reference_energy)
        return {
            "probe_center_of_mass/local_energy_max_abs_error": summary["max_abs_error"],
            "probe_center_of_mass/local_energy_q95_abs_error": summary["q95_abs_error"],
            "probe_center_of_mass/nonfinite_count": summary["nonfinite_count"],
        }


def _pair_distance_positions(
    context: EvaluationContext,
    *,
    r12_min: float,
    r12_max: float,
    n_points: int,
    n_directions: int,
    center_of_mass_radii: Sequence[float],
) -> list[dict[str, Any]]:
    dim = context.batch.spatial_dim
    directions = [direction.to(device=context.batch.device, dtype=context.batch.dtype) for direction in _directions(dim, n_directions)]
    distances = _grid(r12_min, r12_max, n_points, log=True, dtype=context.batch.dtype, device=context.batch.device)
    rows: list[dict[str, Any]] = []
    for center_radius in center_of_mass_radii:
        for direction_id, direction in enumerate(directions):
            center_direction = directions[(direction_id + 1) % len(directions)]
            center = float(center_radius) * center_direction
            for value in distances:
                r12 = float(value.item())
                positions = torch.stack([center + 0.5 * value * direction, center - 0.5 * value * direction])
                rows.append(
                    {
                        "positions": positions,
                        "pair_distance": r12,
                        "center_of_mass_radius": float(center_radius),
                        "direction_id": direction_id,
                    }
                )
    return rows


def _center_of_mass_positions(
    context: EvaluationContext,
    *,
    com_radius_min: float,
    com_radius_max: float,
    n_points: int,
    pair_distance: float,
    n_directions: int,
) -> list[dict[str, Any]]:
    dim = context.batch.spatial_dim
    directions = [direction.to(device=context.batch.device, dtype=context.batch.dtype) for direction in _directions(dim, n_directions)]
    radii = _grid(com_radius_min, com_radius_max, n_points, log=False, dtype=context.batch.dtype, device=context.batch.device)
    pair = torch.tensor(float(pair_distance), dtype=context.batch.dtype, device=context.batch.device)
    rows: list[dict[str, Any]] = []
    for direction_id, direction in enumerate(directions):
        pair_direction = directions[(direction_id + 1) % len(directions)]
        for rho in radii:
            center = rho * direction
            positions = torch.stack([center + 0.5 * pair * pair_direction, center - 0.5 * pair * pair_direction])
            rows.append(
                {
                    "positions": positions,
                    "center_of_mass_radius": float(rho.item()),
                    "pair_distance": float(pair_distance),
                    "direction_id": direction_id,
                }
            )
    return rows


def _evaluate_probe_rows(
    context: EvaluationContext,
    *,
    positions: Sequence[Mapping[str, Any]],
    varying_key: str,
    fixed_key: str,
    reference_energy: float | None,
    include_exact_wavefunction: bool,
    exact_wavefunction: object,
    local_energy_chunk_size: int,
) -> list[dict[str, Any]]:
    if not positions:
        return []
    stacked = torch.stack([row["positions"] for row in positions])
    batch = _batch_like(context, stacked)
    model_result = evaluate_local_energy_in_chunks(
        context.hamiltonian_terms,
        context.model,
        batch,
        return_terms=True,
        chunk_size=local_energy_chunk_size,
    )
    if not isinstance(model_result, LocalEnergyResult):
        raise TypeError("probe local_energy(return_terms=True) must return LocalEnergyResult")
    with torch.no_grad():
        model_output = context.model(batch)
    exact_result = None
    exact_output = None
    if include_exact_wavefunction:
        exact_result = evaluate_local_energy_in_chunks(
            context.hamiltonian_terms,
            exact_wavefunction,
            batch,
            return_terms=True,
            chunk_size=local_energy_chunk_size,
        )
        if not isinstance(exact_result, LocalEnergyResult):
            raise TypeError("exact probe local_energy(return_terms=True) must return LocalEnergyResult")
        with torch.no_grad():
            exact_output = exact_wavefunction(batch)

    rows: list[dict[str, Any]] = []
    for index, meta in enumerate(positions):
        model_energy = model_result.total.detach()[index]
        local_error = "" if reference_energy is None else _finite_float(model_energy - float(reference_energy))
        row: dict[str, Any] = {
            "probe_index": index,
            varying_key: meta[varying_key],
            fixed_key: meta[fixed_key],
            "direction_id": meta["direction_id"],
            "model_logabs": _finite_float(model_output.logabs.detach().reshape(-1)[index]),
            "model_sign": _finite_float(model_output.sign.detach().reshape(-1)[index]),
            "model_relative_abs_psi": "",
            "model_local_energy": _finite_float(model_energy),
            "model_local_energy_error": local_error,
            "kinetic_energy": _term_value(model_result.terms, "kinetic", index),
            "harmonic_trap_energy": _term_value(model_result.terms, "harmonic_trap", index),
            "electron_electron_energy": _term_value(model_result.terms, "electron_electron", index),
            "finite": bool(torch.isfinite(model_energy).item())
            and bool(torch.isfinite(model_output.logabs.detach().reshape(-1)[index]).item()),
        }
        if include_exact_wavefunction and exact_result is not None and exact_output is not None:
            exact_energy = exact_result.total.detach()[index]
            row.update(
                {
                    "exact_logabs": _finite_float(exact_output.logabs.detach().reshape(-1)[index]),
                    "exact_relative_abs_psi": "",
                    "exact_local_energy": _finite_float(exact_energy),
                    "aligned_logabs_error": "",
                    "relative_abs_psi_error": "",
                }
            )
        rows.append(row)

    _fill_relative_amplitudes(rows, include_exact=include_exact_wavefunction)
    return rows


def _fill_relative_amplitudes(rows: list[dict[str, Any]], *, include_exact: bool) -> None:
    model_logabs = [_as_float(row.get("model_logabs")) for row in rows]
    finite_model = [value for value in model_logabs if math.isfinite(value)]
    model_shift = max(finite_model) if finite_model else 0.0
    for row, logabs in zip(rows, model_logabs, strict=True):
        row["model_relative_abs_psi"] = math.exp(logabs - model_shift) if math.isfinite(logabs) else ""

    if not include_exact:
        return
    exact_logabs = [_as_float(row.get("exact_logabs")) for row in rows]
    finite_exact = [value for value in exact_logabs if math.isfinite(value)]
    exact_shift = max(finite_exact) if finite_exact else 0.0
    finite_offsets = [
        model - exact
        for model, exact in zip(model_logabs, exact_logabs, strict=True)
        if math.isfinite(model) and math.isfinite(exact)
    ]
    alignment = _median(finite_offsets) if finite_offsets else 0.0
    for row, model, exact in zip(rows, model_logabs, exact_logabs, strict=True):
        if math.isfinite(exact):
            row["exact_relative_abs_psi"] = math.exp(exact - exact_shift)
        if math.isfinite(model) and math.isfinite(exact):
            row["aligned_logabs_error"] = model - exact - alignment
        model_rel = _as_float(row.get("model_relative_abs_psi"))
        exact_rel = _as_float(row.get("exact_relative_abs_psi"))
        if math.isfinite(model_rel) and math.isfinite(exact_rel):
            row["relative_abs_psi_error"] = model_rel - exact_rel


def _write_probe_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, include_exact: bool) -> None:
    columns = [
        "probe_index",
        "pair_distance",
        "center_of_mass_radius",
        "direction_id",
        "model_logabs",
        "model_sign",
        "model_relative_abs_psi",
        "model_local_energy",
        "model_local_energy_error",
        "kinetic_energy",
        "harmonic_trap_energy",
        "electron_electron_energy",
        "finite",
    ]
    if include_exact:
        columns.extend(
            [
                "exact_logabs",
                "exact_relative_abs_psi",
                "exact_local_energy",
                "aligned_logabs_error",
                "relative_abs_psi_error",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _probe_error_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_energy: float | None,
) -> dict[str, JsonScalar]:
    nonfinite = sum(1 for row in rows if not bool(row.get("finite")))
    if reference_energy is None:
        return {"max_abs_error": None, "q95_abs_error": None, "nonfinite_count": nonfinite}
    errors = sorted(
        abs(_as_float(row.get("model_local_energy_error")))
        for row in rows
        if math.isfinite(_as_float(row.get("model_local_energy_error")))
    )
    return {
        "max_abs_error": max(errors) if errors else None,
        "q95_abs_error": _quantile_sorted(errors, 0.95) if errors else None,
        "nonfinite_count": nonfinite,
    }


def _batch_like(context: EvaluationContext, positions: torch.Tensor) -> ElectronBatch:
    flat = context.batch.flatten_samples()
    spins = None
    if flat.spins is not None:
        spins = flat.spins[:1].repeat(positions.shape[0], 1).to(device=positions.device)
    elif positions.shape[-2] == 2:
        spins = torch.tensor([[1.0, -1.0]], device=positions.device, dtype=positions.dtype).repeat(positions.shape[0], 1)
    return ElectronBatch(
        positions=positions,
        system=flat.system,
        nuclear_positions=flat.nuclear_positions,
        nuclear_charges=flat.nuclear_charges,
        spins=spins,
        aux={},
    )


def _directions(spatial_dim: int, n_directions: int) -> list[torch.Tensor]:
    if spatial_dim <= 0:
        raise ValueError("spatial_dim must be positive")
    base = []
    eye = torch.eye(spatial_dim, dtype=torch.float64)
    for index in range(spatial_dim):
        base.append(eye[index])
    if spatial_dim >= 2:
        vec = torch.zeros(spatial_dim, dtype=torch.float64)
        vec[0] = 1.0
        vec[1] = 1.0
        base.append(vec / torch.linalg.norm(vec))
    if spatial_dim >= 3:
        vec = torch.zeros(spatial_dim, dtype=torch.float64)
        vec[:3] = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        base.append(vec / torch.linalg.norm(vec))
    n = max(1, int(n_directions))
    return [base[index % len(base)] for index in range(n)]


def _grid(
    minimum: float,
    maximum: float,
    n_points: int,
    *,
    log: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    n = max(1, int(n_points))
    if log and minimum > 0.0 and maximum > minimum:
        values = torch.logspace(math.log10(minimum), math.log10(maximum), n, dtype=dtype, device=device)
    else:
        values = torch.linspace(minimum, maximum, n, dtype=dtype, device=device)
    return values


def _resolve_exact_wavefunction(exact_wavefunction: object | str | None) -> object:
    if exact_wavefunction is None:
        return HookeSingletExact()
    if isinstance(exact_wavefunction, str):
        module_name, _, attr = exact_wavefunction.rpartition(".")
        if not module_name or not attr:
            raise ValueError("exact_wavefunction string must be a dotted import path")
        cls = getattr(importlib.import_module(module_name), attr)
        return cls()
    return exact_wavefunction


def _artifact_path(context: EvaluationContext, configured: Path | None, name: str) -> Path:
    if configured is not None:
        return configured
    if context.run_dir is None:
        raise ValueError(f"{name} requires artifact_path when EvaluationContext.run_dir is unavailable")
    return Path(context.run_dir) / "diagnostics" / name / "probe.csv"


def _register_artifact(context: EvaluationContext, *, name: str, path: Path, created_by: str) -> None:
    if context.run_dir is None:
        return
    update_diagnostic_index(
        run_dir=context.run_dir,
        artifacts=[
            {
                "name": name,
                "kind": "csv",
                "path": path,
                "enabled": True,
                "expected": True,
                "created_by": created_by,
            }
        ],
    )


def _register_disabled(context: EvaluationContext, *, name: str, path: Path, created_by: str) -> None:
    if context.run_dir is None:
        return
    update_diagnostic_index(
        run_dir=context.run_dir,
        artifacts=[
            {
                "name": name,
                "kind": "csv",
                "path": path,
                "enabled": False,
                "expected": False,
                "created_by": created_by,
                "warning": "disabled",
            }
        ],
    )


def _term_value(terms: Mapping[str, torch.Tensor], name: str, index: int) -> float | str:
    value = terms.get(name)
    if value is None:
        return ""
    return _finite_float(value.detach().reshape(-1)[index])


def _finite_float(value: torch.Tensor) -> float | str:
    number = float(value.item())
    if math.isfinite(number):
        return number
    return "inf" if number > 0 else "-inf" if number < 0 else "nan"


def _as_float(value: Any) -> float:
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def _quantile_sorted(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float((1.0 - weight) * values[lower] + weight * values[upper])


__all__ = [
    "HookePairCenterOfMassProbe",
    "HookePairDistanceProbe",
]
