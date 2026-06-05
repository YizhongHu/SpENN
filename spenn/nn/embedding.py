"""Embedding scaffold for electron batches."""

from __future__ import annotations

from spenn.data import ElectronBatch, RealFeature
from spenn.nn.equivariant_map import EquivariantMap


class Embedding(EquivariantMap):
    """Base class for embeddings from electron batches to real features."""

    def forward_impl(self, batch: ElectronBatch) -> RealFeature:
        """Embed an electron batch as persistent real tuple features."""

        raise NotImplementedError("Embedding.forward_impl is not implemented yet")


__all__ = ["Embedding"]
