"""Specht representation matrix accessors backed by irrep cache files."""

from __future__ import annotations

import torch

from spenn.data.partition import Partition
from spenn.data.permutation import Permutation
from spenn.reps.irreps import load_default_irrep_metadata


def specht_representation_matrix(
    partition: Partition,
    permutation: Permutation,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a cached orthogonal-basis Specht representation matrix.

    Parameters
    ----------
    partition : Partition
        Specht partition label.
    permutation : Permutation
        Permutation to represent.
    dtype : torch.dtype, optional
        Output dtype.
    device : torch.device, str, or None, optional
        Output device.

    Returns
    -------
    torch.Tensor
        Representation matrix loaded from the checked-in irrep tensor cache.
    """

    return load_default_irrep_metadata().representation_matrix(
        partition,
        permutation,
        dtype=dtype,
        device=device,
    )


__all__ = ["specht_representation_matrix"]
