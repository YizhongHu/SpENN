"""Shared physical Hamiltonian-term metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from tpen.physics.virial import derive_virial_metrics


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
    for name in PHYSICAL_TERM_NAMES:
        tensor = terms.get(name)
        if tensor is None:
            values[name] = None
            continue
        finite = tensor.detach().reshape(-1)
        finite = finite[torch.isfinite(finite)]
        if finite.numel() == 0:
            values[name] = None
            continue
        values[name] = float(finite.mean().item())

    virial = derive_virial_metrics(
        values["kinetic"], values["harmonic_trap"], values["electron_electron"]
    )
    return {
        "virial_residual": virial["residual"],
        "virial_relative_residual": virial["relative_residual"],
    }


__all__ = ["PHYSICAL_TERM_NAMES", "summarize_physical_terms"]
