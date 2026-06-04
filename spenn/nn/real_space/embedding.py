"""Real-space electron embedding scaffold."""

from __future__ import annotations

from typing import Any

from spenn.data.base import EquivariantMap
from spenn.data.batch import ElectronBatch
from spenn.data.real_features import RealFeature


class Embedding(EquivariantMap):
    """Embed electron batches into real-space Specht features.

    Parameters
    ----------
    max_order : int or None, optional
        Maximum feature order requested by the scaffold.
    channels : object or None, optional
        Channel specification reserved for the future implementation.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(self, max_order: int | None = None, channels: object | None = None, **_: Any) -> None:
        super().__init__()
        self.max_order = max_order
        self.channels = channels

    def forward(self, batch: ElectronBatch) -> RealFeature:
        """Return real-space features for an electron batch.

        Parameters
        ----------
        batch : ElectronBatch
            Batched electron coordinates and optional context.

        Returns
        -------
        RealFeature
            Real-space feature blocks.

        Raises
        ------
        NotImplementedError
            Always raised until the real-space embedding math lands.
        """

        raise NotImplementedError("Embedding.forward is not implemented yet")


__all__ = ["Embedding"]
