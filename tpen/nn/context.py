"""Typed forward context shared by TPEN model components."""

from __future__ import annotations

from dataclasses import dataclass

from tpen.data.batch import ElectronBatch
from tpen.dependencies import require_torch
from tpen.nn.basis import ElectronBasisFeatures

torch = require_torch(feature="TPEN forward context")


@dataclass(frozen=True)
class TPENForwardContext:
    """Per-forward typed context for batch-derived model-side quantities.

    Parameters
    ----------
    batch : ElectronBatch
        The physical electron batch for this wavefunction evaluation.
    basis_features : ElectronBasisFeatures or None, optional
        Typed basis output when a basis module is configured.
    """

    batch: ElectronBatch
    basis_features: ElectronBasisFeatures | None = None


__all__ = ["TPENForwardContext"]
