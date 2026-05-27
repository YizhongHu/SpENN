"""Lightweight irrep-aware tensor wrapper and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from spenn.data.partitions import Partition
from spenn.types import Tensor


@dataclass(frozen=True)
class IrrepTensor:
    """Store a tensor with order and Specht irrep metadata.

    Parameters
    ----------
    order : int
        Logical tensor order.
    irrep : Partition
        Specht irrep partition label matching `order`.
    tensor : Tensor
        Tensor carrying the data for the order and irrep.
    """

    order: int
    irrep: Partition
    tensor: Tensor

    def __post_init__(self) -> None:
        if self.irrep.order != self.order:
            raise ValueError(f"IrrepTensor.irrep order mismatch: expected {self.order}, got {self.irrep.order}")

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


def irrep_tail_shape(irrep: Partition) -> tuple[int, int]:
    """Return the final irrep-coordinate axes for a partition.

    Parameters
    ----------
    irrep : Partition
        Specht irrep partition.

    Returns
    -------
    tuple of int
        Expected ``(a, a)`` tail shape. Scalar irreps use ``(1, 1)`` and the
        hard-coded order-3 mixed irrep ``(2, 1)`` uses ``(2, 2)``.
    """

    return (2, 2) if irrep.parts == (2, 1) else (1, 1)


def scalar_channels_last(tensor: torch.Tensor) -> torch.Tensor:
    """Move scalar-irrep feature tensors to channels-last layout.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor with shape ``[batch, channel, n..., 1, 1]``.

    Returns
    -------
    torch.Tensor
        Tensor with scalar irrep axes removed and channel moved to the final
        axis, i.e. ``[batch, n..., channel]``.

    Raises
    ------
    ValueError
        If the final irrep axes are not ``(1, 1)``.
    """

    if tuple(tensor.shape[-2:]) != (1, 1):
        raise ValueError(f"Expected scalar irrep axes (1, 1), got {tuple(tensor.shape[-2:])}")
    return tensor[..., 0, 0].movedim(1, -1)


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
        Tensor to validate with shape ``[batch, channel, n..., a, a]``.
    order : int
        Logical tensor order, equal to the number of particle axes.
    irrep : Partition or None, optional
        Partition metadata for irrep-specific coordinate validation.
    batch_size : int or None, optional
        Expected leading batch size.
    n_electrons : int or None, optional
        Expected size of each ordered particle axis.
    min_channel_dim : int, optional
        Minimum allowed size for the channel axis.

    Raises
    ------
    ValueError
        If `tensor` does not satisfy the expected shape conventions.
    """

    order = int(order)
    if irrep is not None:
        if irrep.order != order:
            raise ValueError(f"irrep order mismatch: expected {order}, got {irrep.order}")
    expected_ndim = order + 4
    if tensor.ndim != expected_ndim:
        raise ValueError(
            f"Expected tensor with shape [batch, channel, n..., a, a] and {expected_ndim} dimensions for order {order}, "
            f"got shape {tuple(tensor.shape)}"
        )
    if batch_size is not None and tensor.shape[0] != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got {tensor.shape[0]}")
    if n_electrons is not None:
        for axis in range(2, 2 + order):
            if tensor.shape[axis] != n_electrons:
                raise ValueError(
                    f"Expected electron axis {axis} to have length {n_electrons}, "
                    f"got {tensor.shape[axis]}"
                )
    if tensor.shape[1] < min_channel_dim:
        raise ValueError(
            f"Expected at least {min_channel_dim} channels in axis 1, "
            f"got {tensor.shape[1]}"
        )
    if irrep is not None:
        expected_tail = irrep_tail_shape(irrep)
        actual_tail = tuple(tensor.shape[-2:])
        if actual_tail != expected_tail:
            raise ValueError(f"Expected trailing irrep axes {expected_tail} for partition {irrep.parts}, got {actual_tail}")
    assert tensor.ndim == expected_ndim


def validate_tensor_product_tensor(
    tensor: torch.Tensor,
    *,
    target: Partition,
    left: Partition,
    right: Partition,
    batch_size: int | None = None,
    n_electrons: int | None = None,
) -> None:
    """Validate tensor-product scaffold tensor shape conventions.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor with shape ``[batch, channel, p, I..., I1..., I2...,
        alpha, beta]``.
    target : Partition
        Target irrep partition.
    left : Partition
        Left source irrep partition.
    right : Partition
        Right source irrep partition.
    batch_size : int or None, optional
        Expected leading batch size.
    n_electrons : int or None, optional
        Expected size of each ordered tuple axis.

    Raises
    ------
    ValueError
        If `tensor` violates the tensor-product convention.
    """

    expected_ndim = 3 + target.order + left.order + right.order + 2
    if tensor.ndim != expected_ndim:
        raise ValueError(
            "Expected TensorProductDict entry with shape [batch, channel, p, "
            "I..., I1..., I2..., alpha, beta], "
            f"got shape {tuple(tensor.shape)}"
        )
    if batch_size is not None and tensor.shape[0] != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got {tensor.shape[0]}")
    if n_electrons is not None:
        first_particle_axis = 3
        n_particle_axes = target.order + left.order + right.order
        for axis in range(first_particle_axis, first_particle_axis + n_particle_axes):
            if tensor.shape[axis] != n_electrons:
                raise ValueError(
                    f"Expected tensor-product particle axis {axis} to have length {n_electrons}, "
                    f"got {tensor.shape[axis]}"
                )
    expected_tail = irrep_tail_shape(target)
    actual_tail = tuple(tensor.shape[-2:])
    if actual_tail != expected_tail:
        raise ValueError(f"Expected trailing target irrep axes {expected_tail} for partition {target.parts}, got {actual_tail}")
    assert tensor.ndim == expected_ndim


def validate_branch_tensor(
    tensor: torch.Tensor,
    *,
    target: Partition,
    source: Partition,
    batch_size: int | None = None,
    n_electrons: int | None = None,
) -> None:
    """Validate branched intermediate tensor shape conventions.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor with shape ``[batch, channel, q, I..., J..., alpha, beta]``.
    target : Partition
        Target feature irrep partition ``lambda``.
    source : Partition
        Source message irrep partition ``mu``.
    batch_size : int or None, optional
        Expected leading batch size.
    n_electrons : int or None, optional
        Expected size of each ordered tuple axis.

    Raises
    ------
    ValueError
        If `tensor` violates the branch intermediate convention.
    """

    expected_ndim = 3 + target.order + source.order + 2
    if tensor.ndim != expected_ndim:
        raise ValueError(
            "Expected BranchDict entry with shape [batch, channel, q, I..., J..., alpha, beta], "
            f"got shape {tuple(tensor.shape)}"
        )
    if batch_size is not None and tensor.shape[0] != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got {tensor.shape[0]}")
    if n_electrons is not None:
        first_particle_axis = 3
        n_particle_axes = target.order + source.order
        for axis in range(first_particle_axis, first_particle_axis + n_particle_axes):
            if tensor.shape[axis] != n_electrons:
                raise ValueError(
                    f"Expected branch particle axis {axis} to have length {n_electrons}, "
                    f"got {tensor.shape[axis]}"
                )
    expected_tail = irrep_tail_shape(target)
    actual_tail = tuple(tensor.shape[-2:])
    if actual_tail != expected_tail:
        raise ValueError(f"Expected trailing target irrep axes {expected_tail} for partition {target.parts}, got {actual_tail}")
    assert tensor.ndim == expected_ndim


__all__ = [
    "IrrepTensor",
    "irrep_tail_shape",
    "scalar_channels_last",
    "validate_branch_tensor",
    "validate_irrep_tensor",
    "validate_tensor_product_tensor",
]
