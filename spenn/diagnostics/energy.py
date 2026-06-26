"""Energy diagnostics for sampled evaluation runs."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from spenn.diagnostics.base import EvaluationContext, JsonScalar
from spenn.diagnostics.artifacts import update_diagnostic_index
from spenn.training.vmc import summarize_local_energy_terms


class EnergyEvaluation:
    """Summarize total and optional per-term local energy for evaluation.

    Parameters
    ----------
    name : str, optional
        Diagnostic name. PR6 emits canonical flat energy metric keys, so this
        name is metadata for the runner and for future collision policy.
    reference_energy : float or None, optional
        Optional exact/reference energy. When provided, signed and absolute
        energy errors are emitted by this evaluation diagnostic.
    include_terms : bool, optional
        Whether to summarize ``EvaluationContext.local_energy_terms``. The
        context must include term energies when this is ``True``.
    quantiles : sequence of float or None, optional
        Local-energy quantiles to report for finite samples.
    include_local_energy_error_quantiles : bool, optional
        Whether to report per-sample local-energy error summaries when
        ``reference_energy`` is provided.
    sampled_eval_table : mapping or None, optional
        Optional bounded sampled-eval table config. Disabled by default.
    """

    def __init__(
        self,
        name: str = "energy",
        reference_energy: float | None = None,
        include_terms: bool = False,
        quantiles: Sequence[float] | None = None,
        include_local_energy_error_quantiles: bool = False,
        sampled_eval_table: Mapping[str, object] | None = None,
        artifact_path: str | Path | None = None,
    ) -> None:
        self.name = str(name)
        self.reference_energy = None if reference_energy is None else float(reference_energy)
        self.include_terms = bool(include_terms)
        self.quantiles = tuple(_validate_quantile(q) for q in (quantiles or (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)))
        self.include_local_energy_error_quantiles = bool(include_local_energy_error_quantiles)
        self.sampled_eval_table = None if sampled_eval_table is None else dict(sampled_eval_table)
        self.artifact_path = None if artifact_path is None else Path(artifact_path)

    def evaluate(self, context: EvaluationContext) -> dict[str, JsonScalar]:
        """Return flat JSON-safe energy metrics."""

        metrics = _summarize_total_energy(context.local_energy, quantiles=self.quantiles)
        if self.reference_energy is not None:
            metrics["reference_energy"] = self.reference_energy
            error = float(metrics["energy"]) - float(self.reference_energy)
            metrics["energy_error"] = error
            metrics["energy_abs_error"] = abs(error)
            if self.include_local_energy_error_quantiles:
                metrics.update(
                    _summarize_local_energy_error(
                        context.local_energy,
                        reference_energy=self.reference_energy,
                        quantiles=self.quantiles,
                    )
                )

        if self.include_terms:
            if context.local_energy_terms is None:
                raise ValueError(
                    "EnergyEvaluation(include_terms=True) requires "
                    "EvaluationContext.local_energy_terms; set Evaluate(return_terms=True)."
                )
            metrics.update(summarize_local_energy_terms(context.local_energy_terms))

        _maybe_write_sampled_eval_table(self, context)
        return metrics


def _summarize_total_energy(
    local_energy: torch.Tensor,
    *,
    quantiles: Sequence[float],
) -> dict[str, JsonScalar]:
    """Summarize finite local-energy samples with VMC-compatible metric names."""

    finite_mask = torch.isfinite(local_energy)
    n_total = int(local_energy.numel())
    n_finite = int(finite_mask.sum().item())

    if n_finite == 0:
        raise ValueError("cannot summarize evaluation energy: no finite local-energy samples")

    finite_energy = local_energy[finite_mask].detach()
    energy = finite_energy.mean()

    if n_finite > 1:
        variance = finite_energy.var(unbiased=False)
    else:
        variance = torch.zeros((), device=finite_energy.device, dtype=finite_energy.dtype)

    std = torch.sqrt(variance)
    stderr = std / float(n_finite) ** 0.5

    metrics: dict[str, JsonScalar] = {
        "energy": float(energy.item()),
        "energy_variance": float(variance.item()),
        "energy_std": float(std.item()),
        "energy_stderr": float(stderr.item()),
        "local_energy_min": float(finite_energy.min().item()),
        "local_energy_max": float(finite_energy.max().item()),
        "local_energy_n_finite": n_finite,
        "local_energy_n_total": n_total,
        "local_energy_finite_fraction": float(n_finite / n_total) if n_total else 0.0,
        "local_energy_nonfinite_count": n_total - n_finite,
    }
    metrics.update(_quantile_metrics("local_energy", finite_energy, quantiles))
    return metrics


def _summarize_local_energy_error(
    local_energy: torch.Tensor,
    *,
    reference_energy: float,
    quantiles: Sequence[float],
) -> dict[str, JsonScalar]:
    finite_mask = torch.isfinite(local_energy)
    finite_error = local_energy[finite_mask].detach() - float(reference_energy)
    metrics: dict[str, JsonScalar] = {
        "local_energy_error_min": float(finite_error.min().item()),
        "local_energy_error_max": float(finite_error.max().item()),
        "local_energy_error_mean": float(finite_error.mean().item()),
        "local_energy_abs_error_mean": float(finite_error.abs().mean().item()),
    }
    metrics.update(_quantile_metrics("local_energy_error", finite_error, quantiles))
    return metrics


def _quantile_metrics(prefix: str, values: torch.Tensor, quantiles: Sequence[float]) -> dict[str, float]:
    q_tensor = torch.tensor(tuple(quantiles), device=values.device, dtype=values.dtype)
    q_values = torch.quantile(values, q_tensor)
    return {
        f"{prefix}_{_quantile_label(q)}": float(value.item())
        for q, value in zip(quantiles, q_values, strict=True)
    }


def _validate_quantile(value: float) -> float:
    q = float(value)
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantiles must lie in [0, 1], got {q}")
    return q


def _quantile_label(value: float) -> str:
    known = {
        0.001: "q001",
        0.01: "q01",
        0.05: "q05",
        0.5: "q50",
        0.95: "q95",
        0.99: "q99",
        0.999: "q999",
    }
    for key, label in known.items():
        if math.isclose(value, key, rel_tol=0.0, abs_tol=1.0e-12):
            return label
    return "q" + f"{100.0 * value:g}".replace(".", "p")


def _maybe_write_sampled_eval_table(diagnostic: EnergyEvaluation, context: EvaluationContext) -> None:
    config = diagnostic.sampled_eval_table
    if config is None:
        return
    enabled = bool(config.get("enabled", False))
    run_dir = context.run_dir
    if run_dir is None:
        return
    run_path = Path(run_dir)
    artifact_path = diagnostic.artifact_path or run_path / "diagnostics" / diagnostic.name / "sampled_eval_table.csv"
    if not enabled:
        update_diagnostic_index(
            run_dir=run_path,
            artifacts=[
                {
                    "name": "sampled_eval_table",
                    "kind": "csv",
                    "path": artifact_path,
                    "enabled": False,
                    "expected": False,
                    "created_by": type(diagnostic).__name__,
                    "warning": "disabled",
                }
            ],
        )
        return
    if context.local_energy_terms is None:
        raise ValueError("sampled_eval_table requires EvaluationContext.local_energy_terms; set return_terms=True")
    max_samples = int(config.get("max_samples", 100000))
    selection = str(config.get("selection", "stride"))
    indices = _selected_indices(int(context.local_energy.numel()), max_samples=max_samples, selection=selection)
    _write_sampled_eval_table(artifact_path, context, indices=indices, reference_energy=diagnostic.reference_energy)
    update_diagnostic_index(
        run_dir=run_path,
        artifacts=[
            {
                "name": "sampled_eval_table",
                "kind": "csv",
                "path": artifact_path,
                "enabled": True,
                "expected": False,
                "created_by": type(diagnostic).__name__,
            }
        ],
    )


def _selected_indices(n_total: int, *, max_samples: int, selection: str) -> list[int]:
    if max_samples <= 0 or n_total <= 0:
        return []
    n_keep = min(n_total, max_samples)
    if selection == "first_n":
        return list(range(n_keep))
    if selection == "stride":
        stride = max(1, math.ceil(n_total / n_keep))
        return list(range(0, n_total, stride))[:n_keep]
    if selection == "seeded_subsample":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        return sorted(torch.randperm(n_total, generator=generator)[:n_keep].tolist())
    raise ValueError(f"unsupported sampled_eval_table selection {selection!r}")


def _write_sampled_eval_table(
    path: Path,
    context: EvaluationContext,
    *,
    indices: Sequence[int],
    reference_energy: float | None,
) -> None:
    flat = context.batch.flatten_samples()
    positions = flat.positions.detach()
    output = context.wavefunction_output
    terms = context.local_energy_terms or {}
    columns = [
        "sample_index",
        "local_energy",
        "local_energy_error",
        "kinetic_energy",
        "harmonic_trap_energy",
        "electron_electron_energy",
        "electron_distance",
        "center_of_mass_radius",
        "radius_e1",
        "radius_e2",
        "position_norm_max",
        "logabs",
        "sign",
        "finite",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for idx in indices:
            pos = positions[idx]
            local_energy = context.local_energy.detach()[idx]
            electron_distance = torch.linalg.norm(pos[0] - pos[1]) if pos.shape[0] >= 2 else torch.tensor(0.0)
            center_of_mass = pos.mean(dim=0)
            row = {
                "sample_index": idx,
                "local_energy": _float_or_text(local_energy),
                "local_energy_error": (
                    ""
                    if reference_energy is None
                    else _float_or_text(local_energy - float(reference_energy))
                ),
                "kinetic_energy": _term_value(terms, "kinetic", idx),
                "harmonic_trap_energy": _term_value(terms, "harmonic_trap", idx),
                "electron_electron_energy": _term_value(terms, "electron_electron", idx),
                "electron_distance": _float_or_text(electron_distance),
                "center_of_mass_radius": _float_or_text(torch.linalg.norm(center_of_mass)),
                "radius_e1": _float_or_text(torch.linalg.norm(pos[0])),
                "radius_e2": _float_or_text(torch.linalg.norm(pos[1])) if pos.shape[0] >= 2 else "",
                "position_norm_max": _float_or_text(torch.linalg.norm(pos, dim=-1).max()),
                "logabs": _float_or_text(output.logabs.detach().reshape(-1)[idx]),
                "sign": _float_or_text(output.sign.detach().reshape(-1)[idx]),
                "finite": bool(torch.isfinite(local_energy).item()),
            }
            writer.writerow(row)


def _term_value(terms: Mapping[str, torch.Tensor], name: str, index: int) -> float | str:
    value = terms.get(name)
    if value is None:
        return ""
    return _float_or_text(value.detach().reshape(-1)[index])


def _float_or_text(value: torch.Tensor) -> float | str:
    number = float(value.item())
    if math.isfinite(number):
        return number
    return "inf" if number > 0 else "-inf" if number < 0 else "nan"


__all__ = ["EnergyEvaluation"]
