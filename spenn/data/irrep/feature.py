"""Path-aggregated irrep tensor feature state."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import torch

from spenn.data.equivariant_state import compare_tensor_mapping
from spenn.data.indices import common_particle_count, permute_tuple_axes
from spenn.data.irrep.base import _normalize_irrep_blocks, _validate_irrep_blocks
from spenn.data.partition import Partition, as_partition
from spenn.data.permutation import Permutation


@dataclass(frozen=True)
class IrrepFeature:
    """Store activated, path-aggregated irrep update blocks.

    Parameters
    ----------
    blocks : mapping
        Mapping ``Partition -> tensor``. The body order is
        ``Partition.order``. Each tensor has shape
        ``[batch, channels, indices..., alpha, beta]``.
    """

    blocks: Mapping[Partition, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = _normalize_irrep_blocks(self.blocks, prefix_ndim=2, name=type(self).__name__)
        object.__setattr__(self, "blocks", normalized)
        self.validate()

    def validate(self) -> "IrrepFeature":
        """Validate tensor types, ranks, tuple axes, and irrep dimensions."""

        _validate_irrep_blocks(self.blocks, prefix_ndim=2, name=type(self).__name__, strict_channels=True)
        return self

    @property
    def n_particles(self) -> int | None:
        """Return the shared particle count if positive-order blocks exist."""

        return common_particle_count(
            self.blocks,
            tuple_axis_start=2,
            order_getter=lambda partition: partition.order,
            name="Irrep tensor blocks",
        )

    def __getitem__(self, partition: Partition) -> torch.Tensor:
        """Return one feature block by partition."""

        return self.blocks[as_partition(partition)]

    def __iter__(self) -> Iterator[Partition]:
        """Iterate over stored partitions."""

        return iter(self.blocks)

    def items(self):
        """Return ``(partition, tensor)`` feature block pairs."""

        return self.blocks.items()

    def clone(self) -> "IrrepFeature":
        """Clone every tensor block."""

        return type(self)({partition: tensor.clone() for partition, tensor in self.blocks.items()})

    def permute(self, permutation: Permutation) -> "IrrepFeature":
        """Return a copy with tuple-index axes permuted."""

        return type(self)(
            {
                partition: permute_tuple_axes(tensor, permutation, axis_start=2, order=partition.order)
                for partition, tensor in self.blocks.items()
            }
        )

    def compare(self, other: "IrrepFeature", *, atol: float = 1.0e-6, rtol: float = 1.0e-6) -> tuple[bool, float]:
        """Compare partition blocks; return ``(is_close, max_abs_error)``."""

        if type(self) is not type(other):
            return False, float("inf")
        return compare_tensor_mapping(self.blocks, other.blocks, atol=atol, rtol=rtol)


__all__ = ["IrrepFeature"]
