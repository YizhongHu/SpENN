"""Final-evaluation symmetry diagnostics."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from spenn.data.batch import ElectronBatch
from spenn.diagnostics.artifacts import update_diagnostic_index
from spenn.diagnostics.base import EvaluationContext, JsonScalar, evaluate_local_energy_in_chunks
from spenn.equivariance.checks import TraceEquivarianceChecker


class PositionExchangeDiagnostic:
    """Check position-only exchange for a symmetric Hooke singlet spatial state."""

    def __init__(
        self,
        name: str = "exchange",
        artifact_path: str | Path | None = None,
        exchange_contract: str = "symmetric_spatial_singlet",
        max_samples: int = 1024,
        enabled: bool = True,
    ) -> None:
        self.name = str(name)
        self.artifact_path = None if artifact_path is None else Path(artifact_path)
        self.exchange_contract = str(exchange_contract)
        self.max_samples = int(max_samples)
        self.enabled = bool(enabled)

    def evaluate(self, context: EvaluationContext) -> Mapping[str, JsonScalar]:
        """Run the position-only exchange check."""

        path = _artifact_path(context, self.artifact_path, self.name, filename="trace.jsonl")
        if not self.enabled:
            _register_artifact(context, name="exchange_trace", path=path, kind="jsonl", created_by=type(self).__name__, enabled=False, expected=False)
            return {}
        if self.exchange_contract != "symmetric_spatial_singlet":
            raise ValueError(f"unsupported exchange_contract {self.exchange_contract!r}")
        batch = _slice_batch(context.batch.flatten_samples(), _selected_indices(context.batch.batch_size, self.max_samples))
        swapped = _swap_positions_only(batch)
        with torch.no_grad():
            original = context.model(batch)
            exchanged = context.model(swapped)
        rows = []
        errors = []
        sign_failures = 0
        nonfinite = 0
        for index in range(batch.batch_size):
            logabs_a = original.logabs.reshape(-1)[index]
            logabs_b = exchanged.logabs.reshape(-1)[index]
            sign_a = original.sign.reshape(-1)[index]
            sign_b = exchanged.sign.reshape(-1)[index]
            finite = bool(torch.isfinite(logabs_a).item() and torch.isfinite(logabs_b).item())
            error = abs(float((logabs_a - logabs_b).item())) if finite else math.inf
            sign_match = bool(sign_a.item() == sign_b.item())
            if finite:
                errors.append(error)
            else:
                nonfinite += 1
            if not sign_match:
                sign_failures += 1
            rows.append(
                {
                    "sample_index": index,
                    "contract": self.exchange_contract,
                    "logabs": _json_number(logabs_a),
                    "exchanged_logabs": _json_number(logabs_b),
                    "logabs_abs_error": error if math.isfinite(error) else "inf",
                    "sign": _json_number(sign_a),
                    "exchanged_sign": _json_number(sign_b),
                    "sign_matches": sign_match,
                    "finite": finite,
                }
            )
        _write_jsonl(path, rows)
        _register_artifact(context, name="exchange_trace", path=path, kind="jsonl", created_by=type(self).__name__)
        return {
            "checks/exchange/logabs_max_abs_error": max(errors) if errors else None,
            "checks/exchange/logabs_mean_abs_error": _mean(errors),
            "checks/exchange/sign_failure_count": sign_failures,
            "checks/exchange/nonfinite_count": nonfinite,
        }


class RotationDiagnostic:
    """Check spatial rotation invariance of log-amplitudes and local energies."""

    def __init__(
        self,
        name: str = "rotation",
        artifact_path: str | Path | None = None,
        max_samples: int = 1024,
        n_rotations: int = 8,
        local_energy_chunk_size: int = 256,
        enabled: bool = True,
    ) -> None:
        self.name = str(name)
        self.artifact_path = None if artifact_path is None else Path(artifact_path)
        self.max_samples = int(max_samples)
        self.n_rotations = int(n_rotations)
        self.local_energy_chunk_size = int(local_energy_chunk_size)
        self.enabled = bool(enabled)

    def evaluate(self, context: EvaluationContext) -> Mapping[str, JsonScalar]:
        """Run the rotation check."""

        path = _artifact_path(context, self.artifact_path, self.name, filename="trace.jsonl")
        if not self.enabled:
            _register_artifact(context, name="rotation_trace", path=path, kind="jsonl", created_by=type(self).__name__, enabled=False, expected=False)
            return {}
        indices = _selected_indices(context.batch.batch_size, self.max_samples)
        batch = _slice_batch(context.batch.flatten_samples(), indices)
        rotations = _rotation_matrices(batch.spatial_dim, self.n_rotations, device=batch.device, dtype=batch.dtype)
        rotated_positions = []
        row_meta = []
        for sample_index in range(batch.batch_size):
            for rotation_id, rotation in enumerate(rotations):
                rotated_positions.append(batch.positions[sample_index] @ rotation.T)
                row_meta.append((sample_index, rotation_id))
        rotated_batch = _replace_positions(batch, torch.stack(rotated_positions))
        with torch.no_grad():
            original_output = context.model(batch)
            rotated_output = context.model(rotated_batch)
        rotated_energy_result = evaluate_local_energy_in_chunks(
            context.hamiltonian_terms,
            context.model,
            rotated_batch,
            return_terms=False,
            chunk_size=self.local_energy_chunk_size,
        )
        if not isinstance(rotated_energy_result, torch.Tensor):
            raise TypeError("rotation local_energy(return_terms=False) must return a torch.Tensor")
        rotated_energy = rotated_energy_result.detach()
        original_energy = context.local_energy.detach().reshape(-1)[indices].to(device=rotated_energy.device, dtype=rotated_energy.dtype)
        rows = []
        logabs_errors = []
        energy_errors = []
        nonfinite = 0
        for row_index, (sample_index, rotation_id) in enumerate(row_meta):
            logabs_a = original_output.logabs.reshape(-1)[sample_index]
            logabs_b = rotated_output.logabs.reshape(-1)[row_index]
            energy_a = original_energy[sample_index]
            energy_b = rotated_energy[row_index]
            finite = bool(
                torch.isfinite(logabs_a).item()
                and torch.isfinite(logabs_b).item()
                and torch.isfinite(energy_a).item()
                and torch.isfinite(energy_b).item()
            )
            logabs_error = abs(float((logabs_b - logabs_a).item())) if finite else math.inf
            energy_error = abs(float((energy_b - energy_a).item())) if finite else math.inf
            if finite:
                logabs_errors.append(logabs_error)
                energy_errors.append(energy_error)
            else:
                nonfinite += 1
            rows.append(
                {
                    "sample_index": int(indices[sample_index]),
                    "rotation_id": rotation_id,
                    "logabs": _json_number(logabs_a),
                    "rotated_logabs": _json_number(logabs_b),
                    "logabs_abs_error": logabs_error if math.isfinite(logabs_error) else "inf",
                    "local_energy": _json_number(energy_a),
                    "rotated_local_energy": _json_number(energy_b),
                    "local_energy_abs_error": energy_error if math.isfinite(energy_error) else "inf",
                    "finite": finite,
                }
            )
        _write_jsonl(path, rows)
        _register_artifact(context, name="rotation_trace", path=path, kind="jsonl", created_by=type(self).__name__)
        return {
            "checks/rotation/logabs_max_abs_error": max(logabs_errors) if logabs_errors else None,
            "checks/rotation/logabs_mean_abs_error": _mean(logabs_errors),
            "checks/rotation/local_energy_max_abs_error": max(energy_errors) if energy_errors else None,
            "checks/rotation/local_energy_mean_abs_error": _mean(energy_errors),
            "checks/rotation/nonfinite_count": nonfinite,
        }


class TraceEquivarianceDiagnostic:
    """Run semantic trace equivariance as a final-evaluation diagnostic."""

    def __init__(
        self,
        name: str = "trace_equivariance",
        artifact_path: str | Path | None = None,
        max_samples: int = 1024,
        permutation_fraction: float = 1.0,
        max_permutations: int = 4,
        seed: int | None = 0,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
        enabled: bool = True,
    ) -> None:
        self.name = str(name)
        self.artifact_path = None if artifact_path is None else Path(artifact_path)
        self.max_samples = int(max_samples)
        self.enabled = bool(enabled)
        self.checker = TraceEquivarianceChecker(
            permutation_fraction=permutation_fraction,
            max_permutations=max_permutations,
            seed=seed,
            compare_output=True,
            dump_on_failure=True,
            atol=atol,
            rtol=rtol,
        )

    def evaluate(self, context: EvaluationContext) -> Mapping[str, JsonScalar]:
        """Run the trace checker and write a summary JSONL artifact."""

        path = _artifact_path(context, self.artifact_path, self.name, filename="trace.jsonl")
        if not self.enabled:
            _register_artifact(context, name="trace_equivariance_trace", path=path, kind="jsonl", created_by=type(self).__name__, enabled=False, expected=False)
            return {}
        indices = _selected_indices(context.batch.batch_size, self.max_samples)
        batch = _slice_batch(context.batch.flatten_samples(), indices)
        state = SimpleNamespace(model=context.model, batch=batch, step=0)
        result = self.checker.run(state)
        artifact = dict(result.artifact or {})
        trace_errors = [
            _as_float(row.get("max_abs_error"))
            for row in artifact.get("trace_errors", [])
            if isinstance(row, Mapping)
        ]
        finite_errors = [value for value in trace_errors if math.isfinite(value)]
        failure_count = int(result.metrics.get("n_failed_entries", 0)) + int(result.metrics.get("n_missing_keys", 0)) + int(result.metrics.get("n_extra_keys", 0))
        row = {
            "check_type": "semantic_trace_equivariance",
            "passed": bool(result.passed),
            "max_abs_error": result.metrics.get("max_abs_error", 0.0),
            "mean_abs_error": _mean(finite_errors) if finite_errors else (0.0 if result.passed else result.metrics.get("max_abs_error", None)),
            "failure_count": failure_count,
            "n_permutations_tested": result.metrics.get("n_permutations_tested", 0),
            "n_trace_entries": result.metrics.get("n_trace_entries", 0),
            "failures": result.failures,
        }
        _write_jsonl(path, [row])
        _register_artifact(context, name="trace_equivariance_trace", path=path, kind="jsonl", created_by=type(self).__name__)
        return {
            "checks/trace_equivariance/max_abs_error": _as_json_scalar(row["max_abs_error"]),
            "checks/trace_equivariance/mean_abs_error": _as_json_scalar(row["mean_abs_error"]),
            "checks/trace_equivariance/failure_count": failure_count,
        }


def _artifact_path(context: EvaluationContext, configured: Path | None, name: str, *, filename: str) -> Path:
    if configured is not None:
        return configured
    if context.run_dir is None:
        raise ValueError(f"{name} requires artifact_path when EvaluationContext.run_dir is unavailable")
    return Path(context.run_dir) / "diagnostics" / name / filename


def _register_artifact(
    context: EvaluationContext,
    *,
    name: str,
    path: Path,
    kind: str,
    created_by: str,
    enabled: bool = True,
    expected: bool = True,
) -> None:
    if context.run_dir is None:
        return
    update_diagnostic_index(
        run_dir=context.run_dir,
        artifacts=[
            {
                "name": name,
                "kind": kind,
                "path": path,
                "enabled": enabled,
                "expected": expected,
                "created_by": created_by,
                "warning": "" if enabled else "disabled",
            }
        ],
    )


def _selected_indices(n_total: int, max_samples: int) -> list[int]:
    if n_total <= 0 or max_samples <= 0:
        return []
    return list(range(min(n_total, max_samples)))


def _slice_batch(batch: ElectronBatch, indices: Sequence[int]) -> ElectronBatch:
    index = torch.as_tensor(indices, device=batch.device, dtype=torch.long)
    positions = batch.positions.index_select(0, index)
    spins = None if batch.spins is None else batch.spins.index_select(0, index)
    return ElectronBatch(
        positions=positions,
        system=batch.system,
        nuclear_positions=batch.nuclear_positions,
        nuclear_charges=batch.nuclear_charges,
        spins=spins,
        aux={},
    )


def _swap_positions_only(batch: ElectronBatch) -> ElectronBatch:
    if batch.n_electrons != 2:
        raise ValueError("PositionExchangeDiagnostic currently requires exactly two electrons")
    positions = batch.positions[:, [1, 0], :]
    return _replace_positions(batch, positions)


def _replace_positions(batch: ElectronBatch, positions: torch.Tensor) -> ElectronBatch:
    spins = batch.spins
    if spins is not None and spins.shape[0] != positions.shape[0]:
        repeat = math.ceil(positions.shape[0] / spins.shape[0])
        spins = spins.repeat(repeat, 1)[: positions.shape[0]]
    return ElectronBatch(
        positions=positions,
        system=batch.system,
        nuclear_positions=batch.nuclear_positions,
        nuclear_charges=batch.nuclear_charges,
        spins=spins,
        aux={},
    )


def _rotation_matrices(
    spatial_dim: int,
    n_rotations: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    if spatial_dim < 2:
        return [torch.eye(spatial_dim, device=device, dtype=dtype)]
    count = max(1, int(n_rotations))
    matrices = []
    for index in range(1, count + 1):
        angle = 2.0 * math.pi * index / (count + 1)
        c = math.cos(angle)
        s = math.sin(angle)
        rotation = torch.eye(spatial_dim, device=device, dtype=dtype)
        rotation[0, 0] = c
        rotation[0, 1] = -s
        rotation[1, 0] = s
        rotation[1, 1] = c
        matrices.append(rotation)
    return matrices


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False))
            handle.write("\n")


def _json_number(value: torch.Tensor) -> float | str:
    number = float(value.item())
    if math.isfinite(number):
        return number
    return "inf" if number > 0 else "-inf" if number < 0 else "nan"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else "nan"
    return value


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _as_float(value: Any) -> float:
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _as_json_scalar(value: Any) -> JsonScalar:
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if math.isfinite(number) else str(value)


__all__ = [
    "PositionExchangeDiagnostic",
    "RotationDiagnostic",
    "TraceEquivarianceDiagnostic",
]
