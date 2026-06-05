"""Irrep tensor states for SpENN.

This module owns partition-keyed irrep tensor containers. Interactions include
a path axis; features and updates are path-aggregated.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import torch

from spenn.data.indices import permute_tuple_axes, validate_tuple_tensor
from spenn.data.partitions import Partition, as_partition
from spenn.data.permutation import Permutation


@dataclass(frozen=True)
class IrrepInteraction:
    """Store path-resolved irrep interaction blocks.

    Parameters
    ----------
    blocks : mapping
        Mapping ``Partition -> tensor``. The body order is
        ``Partition.order``. Each tensor has shape
        ``[batch, channels, paths, indices..., alpha, beta]``.

    Notes
    -----
    The path/channel semantics for interactions are intentionally loose in the
    scaffold. Validation checks tensor rank, shared batch and particle axes,
    and irrep tail dimensions, but same-order channel policies are enforced
    only for :class:`IrrepFeature` and :class:`IrrepUpdate`.
    """

    blocks: Mapping[Partition, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = _normalize_irrep_blocks(self.blocks, prefix_ndim=3, name=type(self).__name__)
        object.__setattr__(self, "blocks", normalized)
        self.validate()

    def validate(self) -> "IrrepInteraction":
        """Validate tensor types, ranks, tuple axes, and irrep dimensions."""

        _validate_irrep_blocks(self.blocks, prefix_ndim=3, name=type(self).__name__, strict_channels=False)
        return self

    @property
    def n_particles(self) -> int | None:
        """Return the shared particle count if positive-order blocks exist."""

        return _common_irrep_particle_count(self.blocks, tuple_axis_start=3)

    def __getitem__(self, partition: Partition) -> torch.Tensor:
        """Return one interaction block by partition."""

        return self.blocks[as_partition(partition)]

    def __iter__(self) -> Iterator[Partition]:
        """Iterate over stored partitions."""

        return iter(self.blocks)

    def items(self):
        """Return ``(partition, tensor)`` interaction block pairs."""

        return self.blocks.items()

    def clone(self) -> "IrrepInteraction":
        """Clone every tensor block."""

        return type(self)({partition: tensor.clone() for partition, tensor in self.blocks.items()})

    def permute(self, permutation: Permutation) -> "IrrepInteraction":
        """Return a copy with tuple-index axes permuted.

        The scaffold applies the particle-label action to tuple axes directly.
        Future Fourier-backed implementations may replace this with an exact
        representation-coordinate action.
        """

        return type(self)(
            {
                partition: permute_tuple_axes(tensor, permutation, axis_start=3, order=partition.order)
                for partition, tensor in self.blocks.items()
            }
        )


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

        return _common_irrep_particle_count(self.blocks, tuple_axis_start=2)

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


@dataclass(frozen=True)
class IrrepUpdate(IrrepFeature):
    """Store irrep-space update proposals.

    `IrrepUpdate` currently has the same layout and behavior as
    :class:`IrrepFeature`; the distinct name marks its role in future
    update-specific maps.
    """


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
    _common_irrep_particle_count(blocks, tuple_axis_start=prefix_ndim)


def _common_irrep_particle_count(
    blocks: Mapping[Partition, torch.Tensor],
    *,
    tuple_axis_start: int,
) -> int | None:
    n_particles: int | None = None
    for partition, tensor in blocks.items():
        if partition.order == 0:
            continue
        current = int(tensor.shape[tuple_axis_start])
        if n_particles is None:
            n_particles = current
        elif current != n_particles:
            raise ValueError(
                "Irrep tensor blocks must share one particle count, "
                f"got {n_particles} and {current}"
            )
    return n_particles


def _irrep_dimension(partition: Partition) -> int:
    from spenn.reps.irreps import irrep_dimension

    return irrep_dimension(partition)


__all__ = ["IrrepFeature", "IrrepInteraction", "IrrepUpdate"]
