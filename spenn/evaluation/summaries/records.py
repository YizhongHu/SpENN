"""Evaluation record writers."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from spenn.evaluation.bundle import EvaluationBundle
from spenn.evaluation.protocols import EvaluationContext
from spenn.evaluation.results import ArtifactRecord, SummaryResult


class SampledRecordWriter:
    """Write a bounded per-sample local-energy table for final eval."""

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
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_index", "local_energy", "logabs", "sign", "finite"],
                lineterminator="\n",
            )
            writer.writeheader()
            for index in indices:
                value = local.local_energy.detach().reshape(-1)[index]
                writer.writerow(
                    {
                        "sample_index": index,
                        "local_energy": _float_or_text(value),
                        "logabs": _float_or_text(wavefunction.logabs.detach().reshape(-1)[index]),
                        "sign": _float_or_text(wavefunction.sign.detach().reshape(-1)[index]),
                        "finite": bool(torch.isfinite(value).item()),
                    }
                )
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


__all__ = ["SampledRecordWriter"]
