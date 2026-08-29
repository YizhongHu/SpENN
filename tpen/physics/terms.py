"""Shared physical Hamiltonian-term metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


PHYSICAL_TERM_NAMES = ("kinetic", "harmonic_trap", "electron_electron")


def summarize_physical_terms(terms: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Summarize Hooke physical terms and their virial diagnostic.

    The virial residual is a diagnostic, not an optimization target:
    ``2 * kinetic - 2 * harmonic_trap + electron_electron``.  It vanishes for
    an exact stationary eigenstate, but need not vanish for a restricted
    variational ansatz that is not stationary under dilation.

    Parameters
    ----------
    terms : Mapping[str, torch.Tensor]
        Per-sample local-energy terms from :class:`LocalEnergyResult`.

    Returns
    -------
    dict
        Mean and population variance for each physical term, followed by the
        virial residual and relative residual. Missing or empty terms produce
        ``None`` for the derived values.
    """

    values: dict[str, float | None] = {}
    variances: dict[str, float | None] = {}
    for name in PHYSICAL_TERM_NAMES:
        tensor = terms.get(name)
        if tensor is None:
            values[name] = None
            variances[name] = None
            continue
        finite = tensor.detach().reshape(-1)
        finite = finite[torch.isfinite(finite)]
        if finite.numel() == 0:
            values[name] = None
            variances[name] = None
            continue
        values[name] = float(finite.mean().item())
        variances[name] = float(finite.var(unbiased=False).item())

    kinetic = values["kinetic"]
    harmonic = values["harmonic_trap"]
    electron_electron = values["electron_electron"]
    if kinetic is None or harmonic is None or electron_electron is None:
        residual = relative = None
    else:
        residual = 2.0 * kinetic - 2.0 * harmonic + electron_electron
        denominator = abs(2.0 * kinetic) + abs(2.0 * harmonic) + abs(electron_electron)
        relative = abs(residual) / denominator if denominator else 0.0

    metrics: dict[str, Any] = {}
    for name in PHYSICAL_TERM_NAMES:
        metrics[f"term/{name}_mean"] = values[name]
        metrics[f"term/{name}_variance"] = variances[name]
    metrics["virial_residual"] = residual
    metrics["virial_relative_residual"] = relative
    return metrics


__all__ = ["PHYSICAL_TERM_NAMES", "summarize_physical_terms"]
