"""Fourier maps between ordered real tensors and Specht coordinates."""

from __future__ import annotations

from collections.abc import Iterable

from spenn.data import IrrepFeature, IrrepInteraction, Partition, RealInteraction, RealUpdate
from spenn.nn.equivariant_map import EquivariantMap


class FourierTransform(EquivariantMap):
    """Map real interactions into path-resolved irrep interactions.

    Parameters
    ----------
    partitions : iterable of Partition or None, optional
        Irrep partitions requested by the transform.
    maps : object or None, optional
        Optional fixed projection maps reserved for generated fixtures.
        Additional keyword arguments are forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        partitions: Iterable[Partition] | None = None,
        maps: object | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.partitions = None if partitions is None else tuple(partitions)
        self.maps = maps

    def forward_impl(self, tensors: RealInteraction) -> IrrepInteraction:
        """Project real interactions to irrep-space interactions.

        Parameters
        ----------
        tensors : RealInteraction
            Path-resolved real-space interaction blocks.

        Returns
        -------
        IrrepInteraction
            Path-resolved irrep-space interaction blocks.

        Raises
        ------
        NotImplementedError
            Always raised until fixed Fourier maps are implemented.
        """

        raise NotImplementedError("FourierTransform.forward_impl is not implemented yet")


class InverseFourierTransform(EquivariantMap):
    """Map path-aggregated irrep features back to real update proposals.

    Parameters
    ----------
    partitions : iterable of Partition or None, optional
        Irrep partitions reconstructed by the transform.
    maps : object or None, optional
        Optional fixed inverse maps reserved for generated fixtures.
        Additional keyword arguments are forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        partitions: Iterable[Partition] | None = None,
        maps: object | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.partitions = None if partitions is None else tuple(partitions)
        self.maps = maps

    def forward_impl(self, tensors: IrrepFeature) -> RealUpdate:
        """Reconstruct real update proposals from irrep features.

        Parameters
        ----------
        tensors : IrrepFeature
            Activated irrep update blocks.

        Returns
        -------
        RealUpdate
            Real-space update proposal blocks.

        Raises
        ------
        NotImplementedError
            Always raised until fixed inverse Fourier maps are implemented.
        """

        raise NotImplementedError("InverseFourierTransform.forward_impl is not implemented yet")


__all__ = [
    "FourierTransform",
    "InverseFourierTransform",
]
