"""Evaluation record writers."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from spenn.evaluation.bundle import EvaluationBundle
from spenn.evaluation.protocols import EvaluationContext
from spenn.evaluation.results import ArtifactRecord, SummaryResult


class SampledRecordWriter:
    """Write a bounded per-sample local-energy table."""

    name = "sampled_records"
    required_fields = frozenset({"local_energy", "wavefunction"})

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_samples: int = 100000,
        filename: str = "sampled_eval_table.csv",
    ) -> None:
        self.enabled = bool(enabled)
        self.max_samples = int(max_samples)
        self.filename = str(filename)

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Write records only when enabled and artifact level permits records."""

        path = context.task_output_dir / self.filename
        if not self.enabled or context.artifact_level != "records":
            return SummaryResult(metrics={})
        local = bundle.local_energy
        wavefunction = bundle.wavefunction
        if local is None or wavefunction is None:
            raise ValueError("SampledRecordWriter requires local_energy and wavefunction")
        flat = bundle.generated.batch.flatten_samples()
        n_total = int(local.local_energy.numel())
        n_keep = min(n_total, max(0, self.max_samples))
        indices = list(range(n_keep))
        base_fields = ["sample_index", "local_energy", "logabs", "sign", "finite"]
        metadata_columns = _metadata_columns(
            bundle.generated.metadata,
            n_total=n_total,
            indices=indices,
            reserved=set(base_fields),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_index",
                    *metadata_columns,
                    "local_energy",
                    "logabs",
                    "sign",
                    "finite",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            for index in indices:
                value = local.local_energy.detach().reshape(-1)[index]
                row = {
                    "sample_index": index,
                    "local_energy": _float_or_text(value),
                    "logabs": _float_or_text(wavefunction.logabs.detach().reshape(-1)[index]),
                    "sign": _float_or_text(wavefunction.sign.detach().reshape(-1)[index]),
                    "finite": bool(torch.isfinite(value).item()),
                }
                for key, values in metadata_columns.items():
                    row[key] = values[index]
                writer.writerow(row)
        return SummaryResult(
            metrics={},
            artifacts=(
                ArtifactRecord(
                    name="sampled_eval_table",
                    kind="csv",
                    path=path,
                    metadata={"rows": len(indices), "n_positions": int(flat.batch_size)},
                ),
            ),
        )


class TransformRecordWriter:
    """Write per-transform comparison records for report plots."""

    name = "transform_records"
    required_fields = frozenset({"transform"})

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_records: int = 100000,
        filename: str = "transform_records.csv",
    ) -> None:
        self.enabled = bool(enabled)
        self.max_records = int(max_records)
        self.filename = str(filename)

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Write transform rows when record artifacts are enabled."""

        del namespace
        if not self.enabled or context.artifact_level != "records":
            return SummaryResult(metrics={})
        transform = bundle.transform
        if transform is None:
            raise ValueError("TransformRecordWriter requires bundle.transform")
        n_total = int(transform.logabs_abs_error.numel())
        n_keep = min(n_total, max(0, self.max_records))
        indices = list(range(n_keep))
        base_fields = [
            "record_index",
            "original_logabs",
            "transformed_logabs",
            "logabs_abs_error",
            "original_sign",
            "transformed_sign",
            "sign_mismatch",
        ]
        include_local_energy = transform.local_energy_abs_error is not None
        if include_local_energy:
            base_fields.append("local_energy_abs_error")
        metadata_columns = _metadata_columns(
            transform.metadata,
            n_total=n_total,
            indices=indices,
            reserved=set(base_fields),
        )
        path = context.task_output_dir / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["record_index", *metadata_columns, *base_fields[1:]],
                lineterminator="\n",
            )
            writer.writeheader()
            for index in indices:
                row: dict[str, object] = {
                    "record_index": index,
                    "original_logabs": _float_or_text(transform.original_logabs.detach().reshape(-1)[index]),
                    "transformed_logabs": _float_or_text(transform.transformed_logabs.detach().reshape(-1)[index]),
                    "logabs_abs_error": _float_or_text(transform.logabs_abs_error.detach().reshape(-1)[index]),
                    "original_sign": _float_or_text(transform.original_sign.detach().reshape(-1)[index]),
                    "transformed_sign": _float_or_text(transform.transformed_sign.detach().reshape(-1)[index]),
                    "sign_mismatch": bool(transform.sign_mismatch.detach().reshape(-1)[index].item()),
                }
                if include_local_energy and transform.local_energy_abs_error is not None:
                    row["local_energy_abs_error"] = _float_or_text(
                        transform.local_energy_abs_error.detach().reshape(-1)[index]
                    )
                for key, values in metadata_columns.items():
                    row[key] = values[index]
                writer.writerow(row)
        return SummaryResult(
            metrics={},
            artifacts=(
                ArtifactRecord(
                    name="transform_records",
                    kind="csv",
                    path=path,
                    metadata={"rows": len(indices), "n_total": n_total},
                ),
            ),
        )


class TraceRecordWriter:
    """Write raw trace, feature-trace, or readout-trace records."""

    name = "trace_records"
    required_fields = frozenset()

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_records: int = 100000,
        filename: str = "trace_records.csv",
    ) -> None:
        self.enabled = bool(enabled)
        self.max_records = int(max_records)
        self.filename = str(filename)

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Write whichever trace-style record collection this task produced."""

        del namespace
        if not self.enabled or context.artifact_level != "records":
            return SummaryResult(metrics={})
        records = _trace_records(bundle)
        if not records:
            return SummaryResult(metrics={})
        rows = [_json_safe_mapping(record) for record in records[: max(0, self.max_records)]]
        columns = sorted({key for row in rows for key in row})
        path = context.task_output_dir / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return SummaryResult(
            metrics={},
            artifacts=(
                ArtifactRecord(
                    name="trace_records",
                    kind="csv",
                    path=path,
                    metadata={"rows": len(rows), "n_total": len(records)},
                ),
            ),
        )


def _float_or_text(value: torch.Tensor) -> float | str:
    number = float(value.item())
    if torch.isfinite(value).item():
        return number
    return "inf" if number > 0 else "-inf" if number < 0 else "nan"


def _metadata_columns(
    metadata: Mapping[str, Any],
    *,
    n_total: int,
    indices: Sequence[int],
    reserved: set[str],
) -> dict[str, dict[int, object]]:
    columns: dict[str, dict[int, object]] = {}
    for raw_key, value in metadata.items():
        key = str(raw_key)
        if key in reserved:
            continue
        column = _metadata_column(value, n_total=n_total)
        if column is None:
            continue
        columns[key] = {index: _csv_value(column[index]) for index in indices}
    return columns


def _metadata_column(value: Any, *, n_total: int) -> Sequence[Any] | None:
    if _is_scalar(value):
        return [value] * n_total
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 0:
            return [tensor.item()] * n_total
        if tensor.ndim == 1 and tensor.shape[0] == n_total:
            return [tensor[index].item() for index in range(n_total)]
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != n_total:
            return None
        if all(_is_scalar(item) for item in value):
            return list(value)
    return None


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _csv_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else "nan"
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 0:
            return _csv_value(tensor.item())
        return json.dumps(tensor.tolist())
    if isinstance(value, Mapping | Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return json.dumps(value)
    return value


def _trace_records(bundle: EvaluationBundle) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    if bundle.trace_comparison is not None:
        records.extend(bundle.trace_comparison.records)
    if bundle.feature_trace is not None:
        records.extend(bundle.feature_trace.records)
    if bundle.readout_trace is not None:
        records.extend(bundle.readout_trace.records)
    return records


def _json_safe_mapping(record: Mapping[str, Any]) -> dict[str, object]:
    return {str(key): _csv_value(value) for key, value in record.items()}


__all__ = ["SampledRecordWriter", "TraceRecordWriter", "TransformRecordWriter"]
