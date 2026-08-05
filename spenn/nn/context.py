"""Typed forward context shared by SpENN model components."""

from __future__ import annotations

from dataclasses import dataclass

from spenn.data.batch import ElectronBatch
from spenn.dependencies import require_torch
from spenn.nn.basis import ElectronBasisFeatures

torch = require_torch(feature="SpENN forward context")


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
