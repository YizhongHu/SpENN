"""Shared helpers for real tuple tensor states."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from spenn.data.indices import common_particle_count, validate_tuple_tensor


def zero_block(
    batch_size: int = 1,
    *,
    paths: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return the reserved zero-order real tensor block.

    Parameters
    ----------
    batch_size : int, optional
        Leading batch dimension.
    paths : int or None, optional
        If provided, include the path axis used by real interactions. ``None``
        returns a feature/update block with shape ``[batch, 0]``.
    device : torch.device, str, or None, optional
        Target device.
    dtype : torch.dtype or None, optional
        Target dtype.

    Returns
    -------
    torch.Tensor
        Empty zero-channel tensor reserved for order 0.
    """

    if batch_size < 0:
        raise ValueError(f"batch_size must be nonnegative, got {batch_size}")
    if paths is None:
        shape = (batch_size, 0)
    else:
        if paths < 0:
            raise ValueError(f"paths must be nonnegative, got {paths}")
        shape = (batch_size, 0, paths)
    return torch.empty(*shape, device=device, dtype=dtype)


def common_real_particle_count(*states: Any) -> int:
    """Return the shared particle count across real tensor states."""

    counts = [state.n_particles for state in states if state.n_particles is not None]
    if not counts:
        raise ValueError("Real tensor states require at least one positive-order block")
    if len(set(counts)) != 1:
        raise ValueError(f"Real tensor particle counts disagree: {counts}")
    return int(counts[0])


def common_real_batch_size(*states: Any) -> int:
    """Return the shared batch size across real tensor states."""

    sizes = [state.batch_size for state in states if state.batch_size is not None]
    if not sizes:
        raise ValueError("Real tensor states require at least one block")
    if len(set(sizes)) != 1:
        raise ValueError(f"Real tensor batch sizes disagree: {sizes}")
    return int(sizes[0])


def common_real_dtype(*states: Any) -> torch.dtype:
    """Return the shared dtype across all blocks in real tensor states."""

    dtypes = {tensor.dtype for state in states for tensor in state.blocks}
    if not dtypes:
        raise ValueError("Real tensor states require at least one block")
    if len(dtypes) != 1:
        raise ValueError(f"Real tensor dtypes disagree: {dtypes}")
    return next(iter(dtypes))


def _normalize_real_blocks(
    blocks: Sequence[torch.Tensor],
    *,
    prefix_ndim: int,
    name: str,
) -> list[torch.Tensor]:
    if isinstance(blocks, torch.Tensor) or not isinstance(blocks, Sequence):
        raise TypeError(f"{name}.blocks must be a sequence of torch.Tensor blocks")
    normalized = list(blocks)
    _validate_real_blocks(normalized, prefix_ndim=prefix_ndim, name=name, strict_zero_channels=True)
    return normalized


def _validate_real_blocks(
    blocks: Sequence[torch.Tensor],
    *,
    prefix_ndim: int,
    name: str,
    strict_zero_channels: bool,
) -> None:
    if isinstance(blocks, torch.Tensor) or not isinstance(blocks, Sequence):
        raise TypeError(f"{name}.blocks must be a sequence of torch.Tensor blocks")
    if not blocks:
        return
    batch_size: int | None = None
    for order, tensor in enumerate(blocks):
        validate_tuple_tensor(tensor, order=order, prefix_ndim=prefix_ndim, name=f"{name}[{order}]")
        if batch_size is None:
            batch_size = int(tensor.shape[0])
        elif int(tensor.shape[0]) != batch_size:
            raise ValueError(f"{name} blocks must share one batch dimension")
    if strict_zero_channels and int(blocks[0].shape[1]) != 0:
        raise ValueError(f"{name}[0] is reserved for zero-order data and must have zero channels")
    common_particle_count(blocks, tuple_axis_start=prefix_ndim)


def _validate_matching_real_blocks(feature: Any, update: Any) -> None:
    if len(feature.blocks) != len(update.blocks):
        raise ValueError("Real feature and update states require matching body-order blocks")
    for order, (feature_block, update_block) in enumerate(zip(feature.blocks, update.blocks)):
        if feature_block.shape != update_block.shape:
            raise ValueError(
                f"Order-{order} feature shape {tuple(feature_block.shape)} does not match "
                f"update shape {tuple(update_block.shape)}"
            )


def _validate_real_update_geometry(feature: Any, update: Any) -> None:
    if len(feature.blocks) != len(update.blocks):
        raise ValueError("Real feature and update states require matching body-order blocks")
    for order, (feature_block, update_block) in enumerate(zip(feature.blocks, update.blocks)):
        if feature_block.shape[0] != update_block.shape[0]:
            raise ValueError(
                f"Order-{order} feature batch {feature_block.shape[0]} does not match "
                f"update batch {update_block.shape[0]}"
            )
        if feature_block.shape[2:] != update_block.shape[2:]:
            raise ValueError(
                f"Order-{order} feature tuple geometry {tuple(feature_block.shape[2:])} does not match "
                f"update tuple geometry {tuple(update_block.shape[2:])}"
            )


__all__ = [
    "common_real_batch_size",
    "common_real_dtype",
    "common_real_particle_count",
    "zero_block",
]
