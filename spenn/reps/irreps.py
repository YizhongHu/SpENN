"""Symmetric-group irrep metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spenn.data.partitions import Partition, as_partition
from spenn.data.permutation import Permutation


@dataclass(frozen=True)
class IrrepMetadata:
    """Store lightweight Specht irrep metadata.

    Parameters
    ----------
    partition : Partition
        Specht partition label.
    dimension : int
        Dimension of the local Specht irrep.
    basis : str, optional
        Basis convention for representation matrices. The scaffold only
        supports ``"orthogonal"``.
    """

    partition: Partition
    dimension: int
    basis: str = "orthogonal"

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
    basis : str, optional
        Basis convention. Only ``"orthogonal"`` is supported for new Specht
        module fixtures.
    """

    def __init__(
        self,
        partition: Partition | tuple[int, ...] | list[int] | str | int,
        dimension: int | None = None,
        *,
        basis: str = "orthogonal",
    ) -> None:
        self.partition = as_partition(partition)
        self.dimension = irrep_dimension(self.partition) if dimension is None else int(dimension)
        if basis != "orthogonal":
            raise ValueError("SpechtIrrep scaffold only supports the orthogonal basis")
        self.basis = basis

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

        return IrrepMetadata(partition=self.partition, dimension=self.dimension, basis=self.basis)

    def representation(self, permutation: Permutation) -> torch.Tensor:
        """Return the representation matrix for `permutation`.

        Parameters
        ----------
        permutation : Permutation
            Permutation to represent.

        Returns
        -------
        torch.Tensor
            Orthogonal-basis representation matrix.

        Raises
        ------
        NotImplementedError
            Always raised until orthogonal-basis Specht matrices are
            implemented.
        """

        raise NotImplementedError("SpechtIrrep.representation is not implemented yet")


def irrep_dimension(partition: Partition) -> int:
    """Return the Specht irrep dimension by the hook-length formula.

    Parameters
    ----------
    partition : Partition
        Specht partition label.

    Returns
    -------
    int
        Dimension used by the existing tensor tail convention.
    """

    if partition.order == 0:
        return 1
    factorial = 1
    for value in range(2, partition.order + 1):
        factorial *= value
    hook_product = 1
    for row, row_length in enumerate(partition.parts):
        for col in range(row_length):
            below = sum(1 for lower in partition.parts[row + 1 :] if lower > col)
            right = row_length - col - 1
            hook_product *= right + below + 1
    return factorial // hook_product


def specht_irrep(
    partition: Partition | tuple[int, ...] | list[int] | str | int,
    *,
    basis: str = "orthogonal",
) -> SpechtIrrep:
    """Construct a :class:`SpechtIrrep` placeholder.

    Parameters
    ----------
    partition : Partition or partition-like
        Specht partition label.
    basis : str, optional
        Basis convention. Only ``"orthogonal"`` is supported.

    Returns
    -------
    SpechtIrrep
        Placeholder irrep object.
    """

    return SpechtIrrep(partition, basis=basis)


__all__ = ["IrrepMetadata", "SpechtIrrep", "irrep_dimension", "specht_irrep"]
