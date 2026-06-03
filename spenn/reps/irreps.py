"""Symmetric-group irrep metadata placeholders."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spenn.data.irrep_tensor import irrep_tail_shape
from spenn.data.partitions import Partition, as_partition
from spenn.reps.real_action import Permutation


@dataclass(frozen=True)
class IrrepMetadata:
    """Store lightweight Specht irrep metadata.

    Parameters
    ----------
    partition : Partition
        Specht partition label.
    dimension : int
        Dimension of the local Specht irrep.
    """

    partition: Partition
    dimension: int

    @property
    def order(self) -> int:
        """Return the partition order.

        Returns
        -------
        int
            Integer partitioned by :attr:`partition`.
        """

        return self.partition.order


class SpechtIrrep:
    """Placeholder symmetric-group irrep object.

    Parameters
    ----------
    partition : Partition or partition-like
        Specht partition label.
    dimension : int or None, optional
        Irrep dimension. If ``None``, use the scaffold tail-shape convention.
    """

    def __init__(self, partition: Partition | tuple[int, ...] | list[int] | str | int, dimension: int | None = None) -> None:
        self.partition = as_partition(partition)
        self.dimension = irrep_dimension(self.partition) if dimension is None else int(dimension)

    @property
    def order(self) -> int:
        """Return the partition order.

        Returns
        -------
        int
            Integer partitioned by :attr:`partition`.
        """

        return self.partition.order

    def metadata(self) -> IrrepMetadata:
        """Return static metadata for this irrep.

        Returns
        -------
        IrrepMetadata
            Partition and dimension metadata.
        """

        return IrrepMetadata(partition=self.partition, dimension=self.dimension)

    def representation(self, permutation: Permutation) -> torch.Tensor:
        """Return the representation matrix for `permutation`.

        Parameters
        ----------
        permutation : Permutation
            Permutation to represent.

        Returns
        -------
        torch.Tensor
            Representation matrix.

        Raises
        ------
        NotImplementedError
            Always raised until real Specht matrices are implemented.
        """

        raise NotImplementedError("SpechtIrrep.representation is not implemented yet")


def irrep_dimension(partition: Partition) -> int:
    """Return the scaffold irrep-coordinate dimension.

    Parameters
    ----------
    partition : Partition
        Specht partition label.

    Returns
    -------
    int
        Dimension used by the existing tensor tail convention.
    """

    return irrep_tail_shape(partition)[0]


def specht_irrep(partition: Partition | tuple[int, ...] | list[int] | str | int) -> SpechtIrrep:
    """Construct a :class:`SpechtIrrep` placeholder.

    Parameters
    ----------
    partition : Partition or partition-like
        Specht partition label.

    Returns
    -------
    SpechtIrrep
        Placeholder irrep object.
    """

    return SpechtIrrep(partition)


__all__ = ["IrrepMetadata", "SpechtIrrep", "irrep_dimension", "specht_irrep"]
