"""Persistent real tuple feature state."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from spenn.data.equivariant_state import compare_tensor_blocks
from spenn.data.indices import common_particle_count, permute_tuple_axes
from spenn.data.permutation import Permutation
from spenn.data.real.base import (
    _normalize_real_blocks,
    _validate_matching_real_blocks,
    _validate_real_blocks,
    _validate_real_update_geometry,
)


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

    def validate_matching_update(self, update: Any) -> Any:
        """Validate that an update has the same block shapes as this feature.

        Parameters
        ----------
        update : object
            Candidate update state with real tuple blocks.

        Returns
        -------
        object
            The validated update, returned for fluent runtime checks.
        """

        _validate_matching_real_blocks(self, update)
        return update

    def validate_update_geometry(self, update: Any) -> Any:
        """Validate an update while allowing channel-count changes.

        Parameters
        ----------
        update : object
            Candidate update state with real tuple blocks.

        Returns
        -------
        object
            The validated update, returned for fluent runtime checks.
        """

        _validate_real_update_geometry(self, update)
        return update

    @property
    def n_particles(self) -> int | None:
        """Return the shared particle count if positive-order blocks exist."""

        return common_particle_count(self.blocks, tuple_axis_start=2)

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

    def compare(self, other: "RealFeature", *, atol: float = 1.0e-6, rtol: float = 1.0e-6) -> tuple[bool, dict[str, float]]:
        """Compare block-by-block; return ``(is_close, max_abs_error)``."""

        if type(self) is not type(other):
            return False, {"max_abs_error": float("inf")}
        return compare_tensor_blocks(self.blocks, other.blocks, atol=atol, rtol=rtol)

    def __add__(self, update: "RealFeature") -> "RealFeature":
        """Return the blockwise sum with another real tuple state."""

        return self.add(update)


__all__ = ["RealFeature"]
