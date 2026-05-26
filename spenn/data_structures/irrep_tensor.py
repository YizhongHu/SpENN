"""Lightweight irrep-aware tensor wrapper and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from spenn.data_structures.partitions import Partition, PartitionLike, normalize_partition
from spenn.types import Tensor


@dataclass(frozen=True)
class IrrepTensor:
    """Store a tensor with order and Specht irrep metadata.

    Parameters
    ----------
    order : int
        Logical tensor order.
    irrep : PartitionLike
        Specht irrep partition label. Inputs are canonicalized to a
        :class:`Partition` matching `order`.
    tensor : Tensor
        Tensor carrying the data for the order and irrep.
    """

    order: int
    irrep: Partition
    tensor: Tensor

    def __post_init__(self) -> None:
        partition = normalize_partition(self.order, self.irrep)
        object.__setattr__(self, "order", partition.order)
        object.__setattr__(self, "irrep", partition)

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "IrrepTensor":
        """Move the wrapped tensor to a new device or dtype.

        Parameters
        ----------
        device : torch.device, str, or None, optional
            Target device. If ``None``, the current device is preserved.
        dtype : torch.dtype or None, optional
            Target dtype. If ``None``, the current dtype is preserved.

        Returns
        -------
        IrrepTensor
            Copy with `tensor` moved to the requested device or dtype.
        """

        return replace(self, tensor=self.tensor.to(device=device, dtype=dtype))


def validate_irrep_tensor(
    tensor: torch.Tensor,
    *,
    order: int,
    irrep: Partition | None = None,
    batch_size: int | None = None,
    n_electrons: int | None = None,
    min_channel_dim: int = 1,
) -> None:
    """Validate irrep-aware tensor shape conventions.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to validate.
    order : int
        Logical tensor order, equal to the number of particle axes.
    irrep : Partition or None, optional
        Partition metadata for irrep-specific coordinate validation.
    batch_size : int or None, optional
        Expected leading batch size.
    n_electrons : int or None, optional
        Expected size of each particle axis.
    min_channel_dim : int, optional
        Minimum allowed size for the channel axis.

    Raises
    ------
    ValueError
        If `tensor` does not satisfy the expected shape conventions.
    """

    if tensor.ndim < order + 2:
        raise ValueError(
            f"Expected tensor with at least {order + 2} dimensions for order {order}, "
            f"got shape {tuple(tensor.shape)}"
        )
    if batch_size is not None and tensor.shape[0] != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got {tensor.shape[0]}")
    if n_electrons is not None:
        for axis in range(1, order + 1):
            if tensor.shape[axis] != n_electrons:
                raise ValueError(
                    f"Expected electron axis {axis} to have length {n_electrons}, "
                    f"got {tensor.shape[axis]}"
                )
    if tensor.shape[order + 1] < min_channel_dim:
        raise ValueError(
            f"Expected at least {min_channel_dim} channels in axis {order + 1}, "
            f"got {tensor.shape[order + 1]}"
        )
    if irrep is not None and order == 3:
        _validate_order3_irrep_shape(tensor, irrep)


def _validate_order3_irrep_shape(tensor: torch.Tensor, irrep: Partition) -> None:
    expected_ndim = 7
    if tensor.ndim != expected_ndim:
        raise ValueError(
            "Expected order-3 tensors with partition metadata to have shape "
            "[batch, n, n, n, channels, irrep_dim, multiplicity_dim], "
            f"got shape {tuple(tensor.shape)}"
        )

    expected_tail = (2, 2) if irrep.parts == (2, 1) else (1, 1)
    actual_tail = tuple(tensor.shape[-2:])
    if actual_tail != expected_tail:
        raise ValueError(f"Expected trailing irrep axes {expected_tail} for partition {irrep.parts}, got {actual_tail}")
