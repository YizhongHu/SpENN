"""Real tuple tensor states for SpENN.

This module owns real-space tuple tensor containers. The semantic classes share
layout validation and permutation logic while keeping distinct names for
persistent features, path-resolved interactions, and update proposals.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch

from spenn.data.indices import permute_tuple_axes, validate_tuple_tensor
from spenn.data.permutation import Permutation


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
        If provided, include the path axis used by :class:`RealInteraction`.
        ``None`` returns a feature/update block with shape ``[batch, 0]``.
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


@dataclass(frozen=True)
class RealFeature:
    """Store persistent real-space tuple feature blocks.

    Parameters
    ----------
    blocks : sequence of torch.Tensor
        Dense list of blocks indexed by body order. Index 0 is reserved for
        the zero-order block and must have shape ``[batch, 0]``. Each positive
        order block has shape ``[batch, channels, indices...]`` with exactly
        `order` tuple-index axes.
    """

    blocks: Sequence[torch.Tensor] = field(default_factory=list)

    def __post_init__(self) -> None:
        blocks = _normalize_real_blocks(self.blocks, prefix_ndim=2, name=type(self).__name__)
        object.__setattr__(self, "blocks", blocks)
        self.validate()

    def validate(self) -> "RealFeature":
        """Validate tensor types, ranks, tuple axes, and batch consistency.

        Returns
        -------
        RealFeature
            The validated state, returned for fluent runtime checks.
        """

        _validate_real_blocks(self.blocks, prefix_ndim=2, name=type(self).__name__, strict_zero_channels=True)
        return self

    @property
    def n_particles(self) -> int | None:
        """Return the shared particle count if positive-order blocks exist."""

        return _common_particle_count(self.blocks, tuple_axis_start=2)

    @property
    def batch_size(self) -> int | None:
        """Return the leading batch size if any block is stored."""

        return None if not self.blocks else int(self.blocks[0].shape[0])

    def __contains__(self, order: int) -> bool:
        """Return whether a block exists for `order`."""

        return 0 <= order < len(self.blocks)

    def __getitem__(self, order: int) -> torch.Tensor:
        """Return one block by body order."""

        return self.blocks[order]

    def __iter__(self) -> Iterator[int]:
        """Iterate over stored body orders."""

        return iter(range(len(self.blocks)))

    def items(self) -> Iterator[tuple[int, torch.Tensor]]:
        """Return ``(order, tensor)`` block pairs."""

        return enumerate(self.blocks)

    def clone(self) -> "RealFeature":
        """Clone every tensor block."""

        return type(self)([tensor.clone() for tensor in self.blocks])

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "RealFeature":
        """Move every block to a new device or dtype."""

        return type(self)([tensor.to(device=device, dtype=dtype) for tensor in self.blocks])

    def permute(self, permutation: Permutation) -> "RealFeature":
        """Return a copy transformed by an active particle permutation."""

        return type(self)(
            [
                permute_tuple_axes(tensor, permutation, axis_start=2, order=order)
                for order, tensor in self.items()
            ]
        )

    def add(self, update: "RealFeature") -> "RealFeature":
        """Return the blockwise sum with another real tuple state."""

        if len(self.blocks) != len(update.blocks):
            raise ValueError("RealFeature.add requires matching body-order blocks")
        return type(self)([left + right for left, right in zip(self.blocks, update.blocks)])

    def __add__(self, update: "RealFeature") -> "RealFeature":
        """Return the blockwise sum with another real tuple state."""

        return self.add(update)


@dataclass(frozen=True)
class RealInteraction:
    """Store path-resolved real-space interaction blocks.

    Parameters
    ----------
    blocks : sequence of torch.Tensor
        Dense list of blocks indexed by target body order. Index 0 is reserved
        for the zero-order block and must have shape ``[batch, 0, paths]``.
        Positive-order blocks have shape
        ``[batch, channels, paths, indices...]``.

    Notes
    -----
    The interaction path/channel semantics are intentionally provisional in
    this scaffold. Validation enforces the shared batch axis, reserved
    zero-order block, and tuple-index geometry, but it does not assign meaning
    to the path axis beyond preserving it under particle permutations.
    """

    blocks: Sequence[torch.Tensor] = field(default_factory=list)

    def __post_init__(self) -> None:
        normalized = _normalize_real_blocks(self.blocks, prefix_ndim=3, name=type(self).__name__)
        object.__setattr__(self, "blocks", normalized)
        self.validate()

    def validate(self) -> "RealInteraction":
        """Validate tensor types, ranks, tuple axes, and batch consistency."""

        _validate_real_blocks(self.blocks, prefix_ndim=3, name=type(self).__name__, strict_zero_channels=True)
        return self

    @property
    def n_particles(self) -> int | None:
        """Return the shared particle count if positive-order blocks exist."""

        return _common_particle_count(self.blocks, tuple_axis_start=3)

    @property
    def batch_size(self) -> int | None:
        """Return the leading batch size if any block is stored."""

        return None if not self.blocks else int(self.blocks[0].shape[0])

    def __getitem__(self, order: int) -> torch.Tensor:
        """Return one interaction block by body order."""

        return self.blocks[order]

    def __iter__(self) -> Iterator[int]:
        """Iterate over stored body orders."""

        return iter(range(len(self.blocks)))

    def items(self) -> Iterator[tuple[int, torch.Tensor]]:
        """Return ``(order, tensor)`` interaction block pairs."""

        return enumerate(self.blocks)

    def clone(self) -> "RealInteraction":
        """Clone every tensor block."""

        return type(self)([tensor.clone() for tensor in self.blocks])

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "RealInteraction":
        """Move every block to a new device or dtype."""

        return type(self)([tensor.to(device=device, dtype=dtype) for tensor in self.blocks])

    def permute(self, permutation: Permutation) -> "RealInteraction":
        """Return a copy transformed by an active particle permutation."""

        return type(self)(
            [
                permute_tuple_axes(tensor, permutation, axis_start=3, order=order)
                for order, tensor in self.items()
            ]
        )


@dataclass(frozen=True)
class RealUpdate(RealFeature):
    """Store real-space tuple update proposal blocks.

    `RealUpdate` has the same tensor layout as :class:`RealFeature`, but its
    semantic role is distinct: it is an update proposal consumed by
    :class:`spenn.nn.Update`.
    """


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
    _common_particle_count(blocks, tuple_axis_start=prefix_ndim)


def _common_particle_count(blocks: Sequence[torch.Tensor], *, tuple_axis_start: int) -> int | None:
    n_particles: int | None = None
    for order, tensor in enumerate(blocks):
        if order == 0:
            continue
        current = int(tensor.shape[tuple_axis_start])
        if n_particles is None:
            n_particles = current
        elif current != n_particles:
            raise ValueError(
                "Tuple tensor blocks must share one particle count, "
                f"got {n_particles} and {current}"
            )
    return n_particles


__all__ = ["RealFeature", "RealInteraction", "RealUpdate", "zero_block"]
