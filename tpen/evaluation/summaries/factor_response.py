"""Artifact and metric summary for common-configuration factor responses."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from tpen.evaluation.bundle import EvaluationBundle, FactorResponseArmValues
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.results import ArtifactRecord, MetricScalar, SummaryResult


class FactorResponseSummary:
    """Write complete aligned arm records and paired response summaries."""

    name = "factor_response"
    required_fields = frozenset({"factor_response"})

    def __init__(self, *, filename: str = "factor_response.csv", max_records: int = 100000) -> None:
        self.filename = str(filename)
        self.max_records = int(max_records)
        if Path(self.filename).name != self.filename:
            raise ValueError("factor-response filename must be one task-local file name")

    def summarize(self, *, bundle: EvaluationBundle, context: EvaluationContext,
                  namespace: str) -> SummaryResult:
        del namespace
        response = bundle.factor_response
        if response is None or response.comparison_kind != "common_configuration":
            raise ValueError("FactorResponseSummary requires common-configuration factor values")
        if not response.model_state_restored:
            raise ValueError("factor-response model restoration was not confirmed")
        flat = bundle.generated.batch.flatten_samples()
        arms = tuple(response.arms)
        n_samples = flat.batch_size
        required = n_samples * len(arms)
        if required > self.max_records:
            raise ValueError("FactorResponseSummary max_records would truncate the common grid: "
                             f"capacity={self.max_records}, required={required}")
        baseline = _arm_by_label(arms, response.baseline_label)
        _validate_arm(baseline, n_samples=n_samples)
        metrics: dict[str, MetricScalar] = {"comparison_is_common_configuration": True,
            "arm_count": len(arms), "configuration_count": n_samples, "model_state_restored": True}
        path = context.task_output_dir / self.filename
        term_names = tuple(baseline.term_energies)
        fields = ["arm", "sample_index", "comparison_kind", "parameter_scales", "realized_parameters",
                  "positions", "local_energy", "delta_local_energy_from_baseline",
                  *[f"term/{name}" for name in term_names], "logabs", "delta_logabs_from_baseline", "sign", "finite"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for arm in arms:
                _validate_arm(arm, n_samples=n_samples, term_names=term_names)
                delta_energy = arm.local_energy.reshape(-1) - baseline.local_energy.reshape(-1)
                delta_logabs = arm.logabs.reshape(-1) - baseline.logabs.reshape(-1)
                finite = torch.isfinite(arm.local_energy.reshape(-1))
                metrics.update(_arm_metrics(arm, delta_energy=delta_energy, delta_logabs=delta_logabs))
                for index in range(n_samples):
                    row: dict[str, object] = {"arm": arm.label, "sample_index": index,
                        "comparison_kind": response.comparison_kind,
                        "parameter_scales": json.dumps(dict(arm.parameter_scales), sort_keys=True),
                        "realized_parameters": json.dumps(dict(arm.realized_parameters), sort_keys=True),
                        "positions": json.dumps(flat.positions[index].detach().cpu().tolist()),
                        "local_energy": float(arm.local_energy.reshape(-1)[index].item()),
                        "delta_local_energy_from_baseline": float(delta_energy[index].item()),
                        "logabs": float(arm.logabs.reshape(-1)[index].item()),
                        "delta_logabs_from_baseline": float(delta_logabs[index].item()),
                        "sign": float(arm.sign.reshape(-1)[index].item()), "finite": bool(finite[index].item())}
                    for name in term_names:
                        row[f"term/{name}"] = float(arm.term_energies[name].reshape(-1)[index].item())
                    writer.writerow(row)
        return SummaryResult(metrics=metrics, artifacts=(ArtifactRecord(
            name="factor_response_common_configuration", kind="csv", path=path,
            metadata={"comparison_kind": response.comparison_kind, "baseline_label": response.baseline_label,
                      "rows": required, "arm_count": len(arms), "configuration_count": n_samples,
                      "selection": "complete_common_configuration_grid", "model_state_restored": True},
        ),))


def _arm_by_label(arms: tuple[FactorResponseArmValues, ...], label: str) -> FactorResponseArmValues:
    matches = [arm for arm in arms if arm.label == label]
    if len(matches) != 1:
        raise ValueError(f"factor response requires exactly one baseline {label!r}")
    return matches[0]


def _validate_arm(arm: FactorResponseArmValues, *, n_samples: int,
                  term_names: tuple[str, ...] | None = None) -> None:
    for name, value in (("local_energy", arm.local_energy), ("logabs", arm.logabs), ("sign", arm.sign)):
        if value.numel() != n_samples:
            raise ValueError(f"factor arm {arm.label!r} {name} is not row-aligned")
    actual_terms = tuple(arm.term_energies)
    if term_names is not None and actual_terms != term_names:
        raise ValueError(f"factor arm {arm.label!r} Hamiltonian terms are not aligned")
    for name, value in arm.term_energies.items():
        if value.numel() != n_samples:
            raise ValueError(f"factor arm {arm.label!r} term {name!r} is not row-aligned")


def _arm_metrics(arm: FactorResponseArmValues, *, delta_energy: torch.Tensor,
                 delta_logabs: torch.Tensor) -> Mapping[str, MetricScalar]:
    prefix = f"arm/{arm.label}"
    energy = arm.local_energy.reshape(-1)
    finite = torch.isfinite(energy)
    finite_energy = energy[finite]
    if finite_energy.numel() == 0:
        raise ValueError(f"factor arm {arm.label!r} has no finite local energies")
    finite_delta_logabs = delta_logabs[torch.isfinite(delta_logabs)]
    if finite_delta_logabs.numel() == 0:
        raise ValueError(f"factor arm {arm.label!r} has no finite log-amplitude responses")
    return {f"{prefix}/local_energy_mean": float(finite_energy.mean().item()),
            f"{prefix}/local_energy_variance": float(finite_energy.var(unbiased=False).item()),
            f"{prefix}/delta_local_energy_mean": float(delta_energy[finite].mean().item()),
            f"{prefix}/delta_logabs_mean": float(finite_delta_logabs.mean().item()),
            f"{prefix}/nonfinite_count": int((~finite).sum().item())}


__all__ = ["FactorResponseSummary"]
