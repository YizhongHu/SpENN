"""Fourier transforms between ordered tuples and Specht coordinates."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from spenn.data.partitions import Partition


class FourierTransform(nn.Module):
    """Base placeholder for fixed Specht Fourier transforms.

    Parameters
    ----------
    partition : Partition or None, optional
        Target Specht partition for the transform.
    maps : object or None, optional
        Optional fixed map payload reserved for generated fixtures.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(self, partition: Partition | None = None, maps: object | None = None, **_: Any) -> None:
        super().__init__()
        self.partition = partition
        self.maps = maps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the fixed Fourier transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to transform.

        Returns
        -------
        torch.Tensor
            Transformed tensor.

        Raises
        ------
        NotImplementedError
            Always raised until generated Fourier fixtures are wired in.
        """

        raise NotImplementedError("FourierTransform.forward is not implemented yet")


class TupleToSpechtFourier(FourierTransform):
    """Placeholder transform from ordered-tuple values to Specht coordinates."""


class SpechtToTupleFourier(FourierTransform):
    """Placeholder transform from Specht coordinates to ordered-tuple values."""


def tuple_to_specht(tensor: torch.Tensor, partition: Partition) -> torch.Tensor:
    """Project ordered-tuple values to Specht coordinates.

    Parameters
    ----------
    tensor : torch.Tensor
        Ordered-tuple tensor.
    partition : Partition
        Target Specht partition.

    Returns
    -------
    torch.Tensor
        Specht-coordinate tensor.

    Raises
    ------
    NotImplementedError
        Always raised until fixed Fourier maps are implemented.
    """

    raise NotImplementedError("tuple_to_specht is not implemented yet")


def specht_to_tuple(tensor: torch.Tensor, partition: Partition) -> torch.Tensor:
    """Reconstruct ordered-tuple values from Specht coordinates.

    Parameters
    ----------
    tensor : torch.Tensor
        Specht-coordinate tensor.
    partition : Partition
        Source Specht partition.

    Returns
    -------
    torch.Tensor
        Ordered-tuple tensor.

    Raises
    ------
    NotImplementedError
        Always raised until fixed Fourier maps are implemented.
    """

    raise NotImplementedError("specht_to_tuple is not implemented yet")


__all__ = [
    "FourierTransform",
    "SpechtToTupleFourier",
    "TupleToSpechtFourier",
    "specht_to_tuple",
    "tuple_to_specht",
]
