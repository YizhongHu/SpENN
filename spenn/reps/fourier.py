"""Fourier maps between ordered real tensors and Specht coordinates."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from spenn.data.base import EquivariantMap
from spenn.data.irrep_features import IrrepTensors
from spenn.data.partitions import Partition
from spenn.data.real_features import RealTensors


class FourierTransform(EquivariantMap):
    """Map real ordered-tuple tensors into temporary irrep coordinates.

    Parameters
    ----------
    partitions : iterable of Partition or None, optional
        Irrep partitions requested by the transform.
    maps : object or None, optional
        Optional fixed projection maps reserved for generated fixtures.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        partitions: Iterable[Partition] | None = None,
        maps: object | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.partitions = None if partitions is None else tuple(partitions)
        self.maps = maps

    def forward(self, tensors: RealTensors) -> IrrepTensors:
        """Project real ordered-tuple tensors to irrep-space tensors.

        Parameters
        ----------
        tensors : RealTensors
            Ordered real-space tensor blocks.

        Returns
        -------
        IrrepTensors
            Temporary irrep-space tensor blocks.

        Raises
        ------
        NotImplementedError
            Always raised until fixed Fourier maps are implemented.
        """

        raise NotImplementedError("FourierTransform.forward is not implemented yet")


class InverseFourierTransform(EquivariantMap):
    """Map temporary irrep coordinates back into real ordered-tuple tensors.

    Parameters
    ----------
    partitions : iterable of Partition or None, optional
        Irrep partitions reconstructed by the transform.
    maps : object or None, optional
        Optional fixed inverse maps reserved for generated fixtures.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        partitions: Iterable[Partition] | None = None,
        maps: object | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.partitions = None if partitions is None else tuple(partitions)
        self.maps = maps

    def forward(self, tensors: IrrepTensors) -> RealTensors:
        """Reconstruct ordered real-space tensors from irrep-space tensors.

        Parameters
        ----------
        tensors : IrrepTensors
            Temporary irrep-space tensor blocks.

        Returns
        -------
        RealTensors
            Ordered real-space tensor blocks.

        Raises
        ------
        NotImplementedError
            Always raised until fixed inverse Fourier maps are implemented.
        """

        raise NotImplementedError("InverseFourierTransform.forward is not implemented yet")


class TupleToSpechtFourier(FourierTransform):
    """Compatibility name for :class:`FourierTransform`."""


class SpechtToTupleFourier(InverseFourierTransform):
    """Compatibility name for :class:`InverseFourierTransform`."""


__all__ = [
    "FourierTransform",
    "InverseFourierTransform",
    "SpechtToTupleFourier",
    "TupleToSpechtFourier",
]
