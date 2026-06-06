"""Shared helpers for irrep tensor states."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from spenn.data.indices import common_particle_count, validate_tuple_tensor
from spenn.data.partition import Partition, as_partition


def _normalize_irrep_blocks(
    blocks: Mapping[Partition, torch.Tensor],
    *,
    prefix_ndim: int,
    name: str,
) -> dict[Partition, torch.Tensor]:
    if not isinstance(blocks, Mapping):
        raise TypeError(f"{name}.blocks must be a mapping from Partition to torch.Tensor")
    normalized: dict[Partition, torch.Tensor] = {}
    for raw_partition, tensor in blocks.items():
        partition = as_partition(raw_partition)
        normalized[partition] = tensor
    _validate_irrep_blocks(normalized, prefix_ndim=prefix_ndim, name=name, strict_channels=False)
    return normalized


def _validate_irrep_blocks(
    blocks: Mapping[Partition, torch.Tensor],
    *,
    prefix_ndim: int,
    name: str,
    strict_channels: bool,
) -> None:
    if not isinstance(blocks, Mapping):
        raise TypeError(f"{name}.blocks must be a mapping from Partition to torch.Tensor")
    batch_size: int | None = None
    channels_by_order: dict[int, int] = {}
    for raw_partition, tensor in blocks.items():
        partition = as_partition(raw_partition)
        order = partition.order
        validate_tuple_tensor(
            tensor,
            order=order,
            prefix_ndim=prefix_ndim,
            suffix_ndim=2,
            name=f"{name}[{partition.parts}]",
        )
        if batch_size is None:
            batch_size = int(tensor.shape[0])
        elif int(tensor.shape[0]) != batch_size:
            raise ValueError(f"{name} blocks must share one batch dimension")
        if strict_channels:
            channels = int(tensor.shape[1])
            previous = channels_by_order.setdefault(order, channels)
            if previous != channels:
                raise ValueError(f"{name} partitions of order {order} must share one channel dimension")
        expected_dim = _irrep_dimension(partition)
        if int(tensor.shape[-2]) != expected_dim or int(tensor.shape[-1]) != expected_dim:
            raise ValueError(
                f"{name}[{partition.parts}] must end with irrep dimensions "
                f"({expected_dim}, {expected_dim}), got {tuple(tensor.shape[-2:])}"
            )
    common_particle_count(
        blocks,
        tuple_axis_start=prefix_ndim,
        order_getter=lambda partition: partition.order,
        name="Irrep tensor blocks",
    )


def _irrep_dimension(partition: Partition) -> int:
    from spenn.reps.irreps import irrep_dimension

    return irrep_dimension(partition)


__all__: list[str] = []
