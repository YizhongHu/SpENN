"""Torch-free virial arithmetic shared by independent study forks."""

from __future__ import annotations

from typing import TypedDict


class VirialMetrics(TypedDict):
    """Scalar virial residual values."""

    residual: float | None
    relative_residual: float | None


def derive_virial_metrics(
    kinetic: float | None,
    harmonic_trap: float | None,
    electron_electron: float | None,
) -> VirialMetrics:
    """Return the Hooke virial residual and relative residual."""

    if kinetic is None or harmonic_trap is None or electron_electron is None:
        return {"residual": None, "relative_residual": None}
    residual = 2.0 * kinetic - 2.0 * harmonic_trap + electron_electron
    denominator = abs(2.0 * kinetic) + abs(2.0 * harmonic_trap) + abs(electron_electron)
    relative = abs(residual) / denominator if denominator else 0.0
    return {"residual": residual, "relative_residual": relative}


__all__ = ["VirialMetrics", "derive_virial_metrics"]
