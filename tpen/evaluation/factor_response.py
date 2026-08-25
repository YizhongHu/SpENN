"""Explicit temporary factor-parameter overrides for evaluation diagnostics."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from tpen.nn.cusp import CurvatureElectronNucleusCuspLaw, ElectronElectronCusp, ElectronNucleusCusp

FACTOR_PARAMETER_NAMES = ("b_ee", "c_electron_nucleus", "d_electron_nucleus")


@dataclass(frozen=True)
class FactorParameterScale:
    """One predeclared physical-parameter scale arm."""

    label: str
    b_ee: float = 1.0
    c_electron_nucleus: float = 1.0
    d_electron_nucleus: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FactorParameterScale":
        """Build one arm while rejecting unknown or invalid identities."""

        allowed = {"label", *FACTOR_PARAMETER_NAMES}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"factor arm carries unknown keys: {unknown}")
        label = str(value.get("label", "")).strip()
        if not label:
            raise ValueError("factor arm requires a non-empty label")
        arm = cls(
            label=label,
            b_ee=float(value.get("b_ee", 1.0)),
            c_electron_nucleus=float(value.get("c_electron_nucleus", 1.0)),
            d_electron_nucleus=float(value.get("d_electron_nucleus", 1.0)),
        )
        for name in FACTOR_PARAMETER_NAMES:
            scale = float(getattr(arm, name))
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"factor arm {label!r} requires finite positive {name}")
        return arm

    def to_dict(self) -> dict[str, float | str]:
        """Return a stable serializable arm identity."""

        return {"label": self.label, "b_ee": self.b_ee,
                "c_electron_nucleus": self.c_electron_nucleus,
                "d_electron_nucleus": self.d_electron_nucleus}


@contextmanager
def helium_factor_parameter_scale(
    model: torch.nn.Module,
    arm: FactorParameterScale | Mapping[str, object],
) -> Iterator[Mapping[str, float]]:
    """Temporarily scale He cusp parameters and byte-verify restoration."""

    resolved = arm if isinstance(arm, FactorParameterScale) else FactorParameterScale.from_mapping(arm)
    ee_factor, en_law = _helium_factor_owners(model)
    parameter_snapshot = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    target_snapshot = {
        "raw_opposite_range": ee_factor.raw_opposite_range.detach().clone(),
        "raw_curvature_coefficient": en_law.raw_curvature_coefficient.detach().clone(),
        "raw_curvature_range": en_law.raw_curvature_range.detach().clone(),
    }
    base = {
        "b_ee": float(ee_factor.opposite_range_parameter.detach().item()),
        "c_electron_nucleus": float(en_law.curvature_coefficient.detach().item()),
        "d_electron_nucleus": float(en_law.curvature_range.detach().item()),
    }
    realized = {name: base[name] * float(getattr(resolved, name)) for name in FACTOR_PARAMETER_NAMES}
    try:
        with torch.no_grad():
            if resolved.b_ee != 1.0:
                ee_factor.raw_opposite_range.copy_(_inverse_positive_parameter(
                    realized["b_ee"], eps=float(ee_factor.eps), like=ee_factor.raw_opposite_range))
            if resolved.c_electron_nucleus != 1.0:
                en_law.raw_curvature_coefficient.copy_(torch.as_tensor(
                    realized["c_electron_nucleus"], device=en_law.raw_curvature_coefficient.device,
                    dtype=en_law.raw_curvature_coefficient.dtype))
            if resolved.d_electron_nucleus != 1.0:
                en_law.raw_curvature_range.copy_(_inverse_positive_parameter(
                    realized["d_electron_nucleus"], eps=float(en_law.eps), like=en_law.raw_curvature_range))
        yield realized
    finally:
        with torch.no_grad():
            ee_factor.raw_opposite_range.copy_(target_snapshot["raw_opposite_range"])
            en_law.raw_curvature_coefficient.copy_(target_snapshot["raw_curvature_coefficient"])
            en_law.raw_curvature_range.copy_(target_snapshot["raw_curvature_range"])
        current = dict(model.named_parameters())
        if tuple(current) != tuple(parameter_snapshot):
            raise RuntimeError("factor response changed the model parameter identity set")
        changed = [name for name, parameter in current.items()
                   if not torch.equal(parameter.detach(), parameter_snapshot[name])]
        if changed:
            raise RuntimeError(f"factor response failed to restore model parameters: {changed}")


def _helium_factor_owners(model: torch.nn.Module) -> tuple[ElectronElectronCusp, CurvatureElectronNucleusCuspLaw]:
    factors = getattr(model, "factors", None)
    if not isinstance(factors, torch.nn.ModuleList):
        raise TypeError("factor response requires model.factors to be a torch.nn.ModuleList")
    ee_factors = [factor for factor in factors if isinstance(factor, ElectronElectronCusp)]
    en_factors = [factor for factor in factors if isinstance(factor, ElectronNucleusCusp)]
    if len(ee_factors) != 1 or len(en_factors) != 1:
        raise ValueError("factor response requires exactly one ElectronElectronCusp and one "
                         f"ElectronNucleusCusp, got {len(ee_factors)} and {len(en_factors)}")
    ee_factor = ee_factors[0]
    en_law = en_factors[0].law
    if not ee_factor.trainable_range:
        raise ValueError("factor response requires a trainable ElectronElectronCusp range")
    if not isinstance(en_law, CurvatureElectronNucleusCuspLaw) or not en_law.trainable:
        raise ValueError("factor response requires a trainable CurvatureElectronNucleusCuspLaw")
    return ee_factor, en_law


def _inverse_positive_parameter(value: float, *, eps: float, like: torch.Tensor) -> torch.Tensor:
    shifted = float(value) - float(eps)
    if not math.isfinite(shifted) or shifted <= 0.0:
        raise ValueError(f"scaled positive factor parameter must exceed eps, got {value}")
    tensor = torch.as_tensor(shifted, device=like.device, dtype=like.dtype)
    return tensor + torch.log(-torch.expm1(-tensor))


__all__ = ["FACTOR_PARAMETER_NAMES", "FactorParameterScale", "helium_factor_parameter_scale"]
