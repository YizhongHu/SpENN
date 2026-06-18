"""Hooke-specific evaluation summaries."""

from __future__ import annotations

import math

import torch

from spenn.evaluation.bundle import DerivativeValues, EvaluationBundle
from spenn.evaluation.protocols import EvaluationContext
from spenn.evaluation.results import MetricScalar, SummaryResult
from spenn.evaluation.summaries.local_energy import summarize_values


class CoalescenceDivergenceSummary:
    """Fit the small-r local-energy ``C_-1 / r`` coefficient."""

    name = "coalescence_divergence"
    required_fields = frozenset({"local_energy"})

    def __init__(self, *, max_fit_points: int | None = None) -> None:
        self.max_fit_points = None if max_fit_points is None else int(max_fit_points)

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return aggregate near-coalescence divergence coefficients."""

        del context, namespace
        local = bundle.local_energy
        if local is None:
            raise ValueError("CoalescenceDivergenceSummary requires local_energy")
        metadata = bundle.generated.metadata
        r12 = _tensor_metadata(metadata, "r12", like=local.local_energy)
        direction_id = _long_metadata(metadata, "direction_id", like=local.local_energy)
        center_id = _optional_long_metadata(metadata, "center_of_mass_id", like=local.local_energy)
        c_values: list[float] = []
        failures = 0
        for key in _group_keys(direction_id, center_id):
            mask = _group_mask(direction_id, center_id, key)
            c_value = _fit_c_minus_one(r12[mask], local.local_energy.reshape(-1)[mask], self.max_fit_points)
            if c_value is None:
                failures += 1
            else:
                c_values.append(abs(c_value))
        if not c_values:
            raise ValueError("no finite coalescence groups available for C_-1 fit")
        values = torch.tensor(c_values, dtype=local.local_energy.dtype, device=local.local_energy.device)
        return SummaryResult(
            metrics={
                "c_minus_1_abs_max": float(values.max().item()),
                "c_minus_1_abs_mean": float(values.mean().item()),
                "c_minus_1_abs_q95": _quantile(values, 0.95),
                "coalescence_fit_finite_fraction": float(len(c_values) / max(1, len(c_values) + failures)),
                "coalescence_fit_failure_count": failures,
            }
        )


class OppositeSpinCuspSummary:
    """Summarize paired-direction opposite-spin cusp slopes."""

    name = "opposite_spin_cusp"
    required_fields = frozenset({"derivatives"})

    def __init__(self, *, derivative_key: str = "r12", expected_slope: float = 0.5) -> None:
        self.derivative_key = str(derivative_key)
        self.expected_slope = float(expected_slope)

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return even/odd paired-direction cusp metrics."""

        del context, namespace
        if bundle.generated.metadata.get("spin_pair", "opposite") != "opposite":
            raise ValueError("OppositeSpinCuspSummary must not be applied to same-spin coalescence")
        values = _derivatives(bundle, self.derivative_key)
        if values.antipodal_pair_id is None or values.direction_sign is None:
            raise ValueError("OppositeSpinCuspSummary requires antipodal_pair_id and direction_sign metadata")
        even_slopes: list[torch.Tensor] = []
        odd_slopes: list[torch.Tensor] = []
        failures = 0
        for pair_id in torch.unique(values.antipodal_pair_id).tolist():
            mask = values.antipodal_pair_id == int(pair_id)
            plus = values.radial_dlogabs[mask & (values.direction_sign > 0)]
            minus = values.radial_dlogabs[mask & (values.direction_sign < 0)]
            if plus.numel() == 0 or minus.numel() == 0:
                failures += 1
                continue
            plus_value = plus[torch.argmin(values.r12[mask & (values.direction_sign > 0)])]
            minus_value = minus[torch.argmin(values.r12[mask & (values.direction_sign < 0)])]
            even_slopes.append(0.5 * (plus_value + minus_value))
            odd_slopes.append(0.5 * (plus_value - minus_value))
        if not even_slopes:
            raise ValueError("no paired opposite-spin cusp directions available")
        even = torch.stack(even_slopes)
        odd = torch.stack(odd_slopes)
        error = torch.abs(even - self.expected_slope)
        return SummaryResult(
            metrics={
                "cusp_even_slope_mean": float(even.mean().item()),
                "cusp_even_slope_abs_error": float(error.mean().item()),
                "cusp_even_slope_abs_error_max": float(error.max().item()),
                "cusp_odd_slant_mean_abs": float(odd.abs().mean().item()),
                "cusp_odd_slant_max_abs": float(odd.abs().max().item()),
                "cusp_pairing_failure_count": failures,
            }
        )


class TailStabilitySummary:
    """Summarize local-energy and logabs pathologies on tail grids."""

    name = "tail_stability"
    required_fields = frozenset({"local_energy", "wavefunction"})

    def __init__(self, *, local_energy_abs_threshold: float | None = None) -> None:
        self.local_energy_abs_threshold = None if local_energy_abs_threshold is None else float(local_energy_abs_threshold)

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return tail-specific scalar metrics."""

        del context, namespace
        local = bundle.local_energy
        wavefunction = bundle.wavefunction
        if local is None or wavefunction is None:
            raise ValueError("TailStabilitySummary requires local_energy and wavefunction")
        energy_metrics = summarize_values(
            local.local_energy,
            quantiles=(0.95, 0.99),
            prefix="local_energy",
        )
        logabs_metrics = summarize_values(
            wavefunction.logabs,
            quantiles=(0.01, 0.99),
            prefix="logabs",
        )
        finite_energy = local.local_energy[torch.isfinite(local.local_energy)]
        threshold = self.local_energy_abs_threshold
        if threshold is None:
            threshold = float(torch.quantile(finite_energy.abs(), torch.tensor(0.99, device=finite_energy.device, dtype=finite_energy.dtype)).item()) if finite_energy.numel() else math.inf
        outliers = int((finite_energy.abs() > threshold).sum().item()) if finite_energy.numel() else 0
        metrics = {**energy_metrics, **logabs_metrics, "tail_outlier_count": outliers}
        return SummaryResult(metrics=metrics)


class PathologyCountSummary:
    """Count nonfinite wavefunction and local-energy pathologies."""

    name = "pathology_count"
    required_fields = frozenset({"local_energy"})

    def __init__(self, *, large_abs_local_energy_threshold: float = 1.0e3) -> None:
        self.large_abs_local_energy_threshold = float(large_abs_local_energy_threshold)

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Return pathology counters."""

        del context, namespace
        local = bundle.local_energy
        if local is None:
            raise ValueError("PathologyCountSummary requires local_energy")
        energy = local.local_energy.detach().reshape(-1)
        nonfinite_energy = ~torch.isfinite(energy)
        metrics: dict[str, MetricScalar] = {
            "nonfinite_local_energy_count": int(nonfinite_energy.sum().item()),
            "large_abs_local_energy_count": int((torch.isfinite(energy) & (energy.abs() > self.large_abs_local_energy_threshold)).sum().item()),
            "finite_fraction": float((~nonfinite_energy).sum().item() / energy.numel()) if energy.numel() else 0.0,
            "pathology_count": int(nonfinite_energy.sum().item()),
        }
        wavefunction = bundle.wavefunction
        if wavefunction is not None:
            logabs = wavefunction.logabs.detach().reshape(-1)
            nonfinite_logabs = ~torch.isfinite(logabs)
            metrics["nonfinite_logabs_count"] = int(nonfinite_logabs.sum().item())
            metrics["pathology_count"] += int(nonfinite_logabs.sum().item())
        else:
            metrics["nonfinite_logabs_count"] = 0
        return SummaryResult(metrics=metrics)


def _derivatives(bundle: EvaluationBundle, key: str) -> DerivativeValues:
    if bundle.derivatives is None or key not in bundle.derivatives:
        raise ValueError(f"missing derivative values for {key!r}")
    return bundle.derivatives[key]


def _tensor_metadata(metadata, key: str, *, like: torch.Tensor) -> torch.Tensor:
    value = metadata.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"metadata field {key!r} must be a tensor")
    return value.to(device=like.device, dtype=like.dtype).reshape(-1)


def _long_metadata(metadata, key: str, *, like: torch.Tensor) -> torch.Tensor:
    value = metadata.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"metadata field {key!r} must be a tensor")
    return value.to(device=like.device, dtype=torch.long).reshape(-1)


def _optional_long_metadata(metadata, key: str, *, like: torch.Tensor) -> torch.Tensor | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"metadata field {key!r} must be a tensor")
    return value.to(device=like.device, dtype=torch.long).reshape(-1)


def _group_keys(direction_id: torch.Tensor, center_id: torch.Tensor | None) -> set[tuple[int, int]]:
    if center_id is None:
        return {(0, int(value)) for value in torch.unique(direction_id).tolist()}
    return {(int(c), int(d)) for c, d in zip(center_id.tolist(), direction_id.tolist(), strict=True)}


def _group_mask(direction_id: torch.Tensor, center_id: torch.Tensor | None, key: tuple[int, int]) -> torch.Tensor:
    center_key, direction_key = key
    mask = direction_id == direction_key
    if center_id is not None:
        mask = mask & (center_id == center_key)
    return mask


def _fit_c_minus_one(r12: torch.Tensor, energy: torch.Tensor, max_fit_points: int | None) -> float | None:
    finite = torch.isfinite(r12) & torch.isfinite(energy) & (r12 > 0)
    if int(finite.sum().item()) < 3:
        return None
    x = r12[finite]
    y = energy[finite]
    order = torch.argsort(x)
    if max_fit_points is not None and max_fit_points > 0:
        order = order[:max_fit_points]
    if int(order.numel()) < 3:
        return None
    x = x[order]
    y = y[order]
    design = torch.stack([1.0 / x, torch.ones_like(x), x], dim=-1)
    solution = torch.linalg.lstsq(design, y.unsqueeze(-1)).solution
    return float(solution[0, 0].item())


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return math.nan
    return float(torch.quantile(values, torch.tensor(q, device=values.device, dtype=values.dtype)).item())


__all__ = [
    "CoalescenceDivergenceSummary",
    "OppositeSpinCuspSummary",
    "PathologyCountSummary",
    "TailStabilitySummary",
]
