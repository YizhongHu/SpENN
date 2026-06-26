"""Typed forward context shared by SpENN model components."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field

from spenn.data.batch import ElectronBatch
from spenn.dependencies import require_torch
from spenn.nn.basis import ElectronBasisFeatures

torch = require_torch(feature="SpENN forward context")


@dataclass(frozen=True)
class SpENNForwardContext:
    """Per-forward typed context for batch-derived model-side quantities.

    Parameters
    ----------
    batch : ElectronBatch
        The physical electron batch for this wavefunction evaluation.
    basis_features : ElectronBasisFeatures or None, optional
        Typed basis output when a basis module is configured.
    coordinate_envelopes : mutable mapping of str to torch.Tensor, optional
        Cached invariant coordinate scalars or gates derived from ``batch``.
    """

    batch: ElectronBatch
    basis_features: ElectronBasisFeatures | None = None
    coordinate_envelopes: MutableMapping[str, torch.Tensor] = field(default_factory=dict)

    def coordinate_envelope(self, key: str) -> torch.Tensor | None:
        """Return a cached coordinate envelope/scalar by key if present."""

        return self.coordinate_envelopes.get(key)


__all__ = ["SpENNForwardContext"]
