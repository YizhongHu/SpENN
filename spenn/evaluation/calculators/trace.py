"""Trace-based evaluation calculators."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.data.equivariant_state import apply_particle_permutation
from spenn.data.permutation import Permutation
from spenn.evaluation.bundle import (
    EvaluationBundle,
    FeatureTraceValues,
    ReadoutTraceValues,
    TraceComparisonValues,
)
from spenn.evaluation.calculators.transforms import split_paired_batch
from spenn.evaluation.protocols import EvaluationContext
from spenn.trace import ParticleTensor, Trace


class TraceEquivarianceCalculator:
    """Compare matching trace entries across permutation orbits."""

    name = "trace_equivariance"

    def __init__(
        self,
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
        compare_slots: Sequence[str] | None = None,
    ) -> None:
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.compare_slots = None if compare_slots is None else frozenset(str(slot) for slot in compare_slots)

    def calculate(
        self,
        *,
        model: torch.nn.Module,
        bundle: EvaluationBundle,
        context: EvaluationContext,
    ) -> EvaluationBundle:
        """Capture paired traces and store raw typed comparison records."""

        del context
        original, transformed = split_paired_batch(bundle.generated.batch)
        permutation_images = _metadata_permutations(bundle.generated.metadata, n_rows=original.batch_size, device=original.device)
        records: list[dict[str, Any]] = []
        errors: list[float] = []
        missing_count = 0
        extra_count = 0
        failure_count = 0
        compared_entry_count = 0
        comparison_error_count = 0
        for image, indices in _permutation_groups(permutation_images).items():
            permutation = Permutation(image)
            original_group = _select_batch(original, indices)
            transformed_group = _select_batch(transformed, indices)
            with torch.no_grad():
                with Trace.capture(model=model) as trace_a:
                    model(original_group)
                with Trace.capture(model=model) as trace_b:
                    model(transformed_group)
            entries_a = _filtered_entries(trace_a, self.compare_slots)
            entries_b = _filtered_entries(trace_b, self.compare_slots)
            keys_a = set(entries_a)
            keys_b = set(entries_b)
            missing = sorted(keys_a - keys_b)
            extra = sorted(keys_b - keys_a)
            missing_count += len(missing)
            extra_count += len(extra)
            failure_count += len(missing) + len(extra)
            for key in missing:
                records.append({"key": key, "status": "missing", "permutation": image})
            for key in extra:
                records.append({"key": key, "status": "extra", "permutation": image})
            for key in sorted(keys_a & keys_b):
                compared_entry_count += 1
                try:
                    expected = apply_particle_permutation(entries_a[key].value, permutation)
                    actual = entries_b[key].value
                    compare = getattr(actual, "compare", None)
                    if not callable(compare):
                        raise TypeError(f"trace value {type(actual).__name__} has no compare(...) contract")
                    close, metrics = compare(expected, atol=self.atol, rtol=self.rtol)
                    error = float(metrics.get("max_abs_error", 0.0))
                except Exception as exc:
                    close = False
                    error = math.inf
                    comparison_error_count += 1
                    records.append(
                        {
                            "key": key,
                            "status": "error",
                            "permutation": image,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                else:
                    records.append(
                        {
                            "key": key,
                            "status": "success" if close else "failed",
                            "permutation": image,
                            "max_abs_error": error,
                        }
                    )
                if math.isfinite(error):
                    errors.append(error)
                if not close:
                    failure_count += 1
        device = original.device
        dtype = original.dtype
        error_tensor = torch.tensor(errors, device=device, dtype=dtype)
        max_error = error_tensor if error_tensor.numel() else torch.zeros(0, device=device, dtype=dtype)
        mean_error = error_tensor if error_tensor.numel() else torch.zeros(0, device=device, dtype=dtype)
        return replace(
            bundle,
            trace_comparison=TraceComparisonValues(
                max_abs_error=max_error,
                mean_abs_error=mean_error,
                failure_count=failure_count,
                compared_entry_count=compared_entry_count,
                comparison_error_count=comparison_error_count,
                missing_key_count=missing_count,
                extra_key_count=extra_count,
                records=tuple(records),
            ),
        )


class FeatureTraceCalculator:
    """Collect feature-like trace-entry magnitude records."""

    name = "feature_trace"

    def __init__(
        self,
        *,
        slots: Sequence[str] | None = None,
        norm_ord: int | float = 2,
    ) -> None:
        self.slots = None if slots is None else frozenset(str(slot) for slot in slots)
        self.norm_ord = norm_ord

    def calculate(
        self,
        *,
        model: torch.nn.Module,
        bundle: EvaluationBundle,
        context: EvaluationContext,
    ) -> EvaluationBundle:
        """Capture feature trace entries and store raw magnitude records."""

        del context
        batch = bundle.generated.batch.flatten_samples()
        with torch.no_grad():
            with Trace.capture(model=model) as trace:
                model(batch)
        records = []
        for entry in trace:
            if self.slots is not None and entry.slot not in self.slots:
                continue
            if not _is_feature_entry(entry):
                continue
            tensors = _entry_tensors(entry.value)
            if not tensors:
                continue
            records.append(_magnitude_record(entry, tensors))
        return replace(bundle, feature_trace=FeatureTraceValues(records=tuple(records)))


class ReadoutTraceCalculator:
    """Collect readout/Pfaffian conditioning records."""

    name = "readout_trace"

    def __init__(
        self,
        *,
        slots: Sequence[str] | None = None,
        singular_value_eps: float = 1.0e-8,
    ) -> None:
        self.slots = None if slots is None else frozenset(str(slot) for slot in slots)
        self.singular_value_eps = float(singular_value_eps)

    def calculate(
        self,
        *,
        model: torch.nn.Module,
        bundle: EvaluationBundle,
        context: EvaluationContext,
    ) -> EvaluationBundle:
        """Capture readout-like matrices and store conditioning records."""

        del context
        batch = bundle.generated.batch.flatten_samples()
        with torch.no_grad():
            with Trace.capture(model=model) as trace:
                output = model(batch)
        records: list[dict[str, Any]] = []
        if isinstance(output, WavefunctionOutput):
            matrix = output.aux.get("K")
            if isinstance(matrix, torch.Tensor):
                records.extend(_matrix_records("wavefunction_output/K", matrix, eps=self.singular_value_eps))
        for entry in trace:
            if self.slots is not None and entry.slot not in self.slots:
                continue
            if not _is_readout_entry(entry):
                continue
            for tensor in _entry_tensors(entry.value):
                records.extend(_matrix_records(entry.key, tensor, eps=self.singular_value_eps))
        return replace(bundle, readout_trace=ReadoutTraceValues(records=tuple(records)))


def _metadata_permutations(metadata: Mapping[str, object], *, n_rows: int, device: torch.device) -> torch.Tensor:
    value = metadata.get("permutation")
    if not isinstance(value, torch.Tensor):
        raise ValueError("TraceEquivarianceCalculator requires tensor metadata 'permutation'")
    images = value.to(device=device, dtype=torch.long)
    if images.ndim != 2 or images.shape[0] != n_rows:
        raise ValueError(f"permutation metadata must have shape [batch, n_particles], got {tuple(images.shape)}")
    return images


def _permutation_groups(images: torch.Tensor) -> dict[tuple[int, ...], torch.Tensor]:
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, image in enumerate(images.detach().cpu().tolist()):
        groups.setdefault(tuple(int(item) for item in image), []).append(index)
    return {
        image: torch.tensor(indices, device=images.device, dtype=torch.long)
        for image, indices in groups.items()
    }


def _select_batch(batch: ElectronBatch, indices: torch.Tensor) -> ElectronBatch:
    return ElectronBatch(
        positions=batch.positions.index_select(0, indices),
        system=batch.system,
        nuclear_positions=_select_optional(batch.nuclear_positions, indices, batch_size=batch.batch_size),
        nuclear_charges=_select_optional(batch.nuclear_charges, indices, batch_size=batch.batch_size),
        spins=None if batch.spins is None else batch.spins.index_select(0, indices),
        aux={},
    )


def _select_optional(value: torch.Tensor | None, indices: torch.Tensor, *, batch_size: int) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim >= 1 and value.shape[0] == batch_size:
        return value.index_select(0, indices)
    return value


def _filtered_entries(trace: Trace, slots: frozenset[str] | None) -> dict[str, Any]:
    return {
        entry.key: entry
        for entry in trace
        if slots is None or entry.slot in slots
    }


def _is_feature_entry(entry: Any) -> bool:
    if entry.semantic_type in {"features", "irrep_features", "pair_features"}:
        return True
    name = type(entry.value).__name__
    return "Feature" in name or entry.slot in {"feature", "features", "interaction", "output"}


def _is_readout_entry(entry: Any) -> bool:
    if entry.semantic_type in {"readout_matrix", "pfaffian_matrix", "logabs_component"}:
        return True
    text = f"{entry.key}/{entry.slot}/{type(entry.value).__name__}".lower()
    return any(token in text for token in ("readout", "pfaffian", "matrix"))


def _entry_tensors(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, ParticleTensor):
        return (value.value,)
    if isinstance(value, torch.Tensor):
        return (value,)
    blocks = getattr(value, "blocks", None)
    if isinstance(blocks, Mapping):
        tensors = tuple(block for block in blocks.values() if isinstance(block, torch.Tensor))
        return tensors
    if isinstance(blocks, Sequence):
        tensors = tuple(block for block in blocks if isinstance(block, torch.Tensor))
        return tensors
    return ()


def _magnitude_record(entry: Any, tensors: Sequence[torch.Tensor]) -> dict[str, Any]:
    flat = _flatten_tensors(tensors)
    finite = flat[torch.isfinite(flat)]
    nonfinite_count = int(flat.numel() - finite.numel())
    if finite.numel() == 0:
        return {
            "entry_key": entry.key,
            "slot": entry.slot,
            "producer_name": entry.producer_name,
            "shape": tuple(tuple(tensor.shape) for tensor in tensors),
            "rms": math.inf,
            "max_abs": math.inf,
            "q95_abs": math.inf,
            "finite_fraction": 0.0,
            "nonfinite_count": nonfinite_count,
        }
    abs_values = finite.abs()
    return {
        "entry_key": entry.key,
        "slot": entry.slot,
        "producer_name": entry.producer_name,
        "shape": tuple(tuple(tensor.shape) for tensor in tensors),
        "rms": float(torch.sqrt(torch.mean(finite.pow(2))).item()),
        "max_abs": float(abs_values.max().item()),
        "q95_abs": float(torch.quantile(abs_values, torch.tensor(0.95, device=abs_values.device, dtype=abs_values.dtype)).item()),
        "finite_fraction": float(finite.numel() / flat.numel()) if flat.numel() else 0.0,
        "nonfinite_count": nonfinite_count,
    }


def _matrix_records(key: str, tensor: torch.Tensor, *, eps: float) -> list[dict[str, Any]]:
    if tensor.ndim < 2 or tensor.shape[-1] != tensor.shape[-2]:
        return []
    matrices = tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])
    records = []
    for index, matrix in enumerate(matrices):
        finite_fraction = float(torch.isfinite(matrix).sum().item() / matrix.numel()) if matrix.numel() else 0.0
        if finite_fraction < 1.0:
            records.append(
                {
                    "entry_key": key,
                    "matrix_index": index,
                    "matrix_shape": tuple(matrix.shape),
                    "s_min": math.nan,
                    "s_max": math.nan,
                    "condition_number": math.inf,
                    "near_zero_count": int((~torch.isfinite(matrix)).sum().item()),
                    "finite_fraction": finite_fraction,
                }
            )
            continue
        singular_values = torch.linalg.svdvals(matrix)
        s_min = float(singular_values.min().item()) if singular_values.numel() else math.nan
        s_max = float(singular_values.max().item()) if singular_values.numel() else math.nan
        condition = math.inf if s_min <= eps else s_max / s_min
        records.append(
            {
                "entry_key": key,
                "matrix_index": index,
                "matrix_shape": tuple(matrix.shape),
                "s_min": s_min,
                "s_max": s_max,
                "condition_number": condition,
                "near_zero_count": int((singular_values <= eps).sum().item()),
                "finite_fraction": finite_fraction,
            }
        )
    return records


def _flatten_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.empty(0)
    return torch.cat([tensor.detach().reshape(-1) for tensor in tensors], dim=0)


__all__ = [
    "FeatureTraceCalculator",
    "ReadoutTraceCalculator",
    "TraceEquivarianceCalculator",
]
