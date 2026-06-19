"""Evaluation record writers."""

from __future__ import annotations

import csv
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
    return value


__all__ = ["SampledRecordWriter"]
