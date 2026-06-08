"""Fourier maps between ordered real tensors and Specht coordinates."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch

from spenn.data.indices import permute_tuple_slots
from spenn.data.irrep import IrrepFeature, IrrepInteraction
from spenn.data.partition import Partition, integer_partitions
from spenn.data.permutation import all_permutations
from spenn.data.real import RealInteraction, RealUpdate, zero_block
from spenn.equivariance import EquivariantMap
from spenn.reps.irreps import IrrepMetadata, irrep_dimension, load_default_irrep_metadata


class FourierTransform(EquivariantMap):
    """Map real interactions into path-resolved irrep interactions.

    Parameters
    ----------
    partitions : iterable of Partition or None, optional
        Irrep partitions requested by the transform.
    metadata : IrrepMetadata, path-like, or None, optional
        Irrep metadata carrying Sage-generated representation matrices. If
        ``None``, checked-in cache files are loaded.
    Additional keyword arguments are forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        partitions: Iterable[Partition] | None = None,
        metadata: IrrepMetadata | str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.partitions = None if partitions is None else tuple(partitions)
        self.metadata = _coerce_metadata(metadata)

    def forward_impl(self, tensors: RealInteraction) -> IrrepInteraction:
        """Project real interactions to irrep-space interactions.

        Parameters
        ----------
        tensors : RealInteraction
            Path-resolved real-space interaction blocks.

        Returns
        -------
        IrrepInteraction
            Path-resolved irrep-space interaction blocks.

        """

        tensors.validate()
        blocks = {}
        for order, tensor in tensors.items():
            if order == 0 or tensor.shape[1] == 0:
                continue
            for partition in self._partitions_for_order(order):
                blocks[partition] = _fourier_block(
                    tensor,
                    partition,
                    tuple_axis_start=3,
                    metadata=self.metadata,
                )
        return IrrepInteraction(blocks)

    def _partitions_for_order(self, order: int) -> tuple[Partition, ...]:
        if self.partitions is not None:
            return tuple(partition for partition in self.partitions if partition.order == order)
        return integer_partitions(order)


class InverseFourierTransform(EquivariantMap):
    """Map path-aggregated irrep features back to real update proposals.

    Parameters
    ----------
    partitions : iterable of Partition or None, optional
        Irrep partitions reconstructed by the transform.
    metadata : IrrepMetadata, path-like, or None, optional
        Irrep metadata carrying Sage-generated representation matrices. If
        ``None``, checked-in cache files are loaded.
    Additional keyword arguments are forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        partitions: Iterable[Partition] | None = None,
        metadata: IrrepMetadata | str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.partitions = None if partitions is None else tuple(partitions)
        self.metadata = _coerce_metadata(metadata)

    def forward_impl(self, tensors: IrrepFeature) -> RealUpdate:
        """Reconstruct real update proposals from irrep features.

        Parameters
        ----------
        tensors : IrrepFeature
            Activated irrep update blocks.

        Returns
        -------
        RealUpdate
            Real-space update proposal blocks.

        """

        tensors.validate()
        if not tensors.blocks:
            return RealUpdate([])
        max_order = max(partition.order for partition in tensors.blocks)
        batch_size = next(iter(tensors.blocks.values())).shape[0]
        first = next(iter(tensors.blocks.values()))
        output: list[torch.Tensor] = [zero_block(batch_size=batch_size, device=first.device, dtype=first.dtype)]
        for order in range(1, max_order + 1):
            projected = []
            for partition, tensor in tensors.items():
                if partition.order != order:
                    continue
                projected.append(
                    _inverse_fourier_block(
                        tensor,
                        partition,
                    )
                )
            if not projected:
                n_particles = tensors.n_particles
                if n_particles is None:
                    raise ValueError(f"Cannot infer tuple axes for missing order-{order} inverse Fourier block")
                output.append(torch.empty(batch_size, 0, *((n_particles,) * order), device=first.device, dtype=first.dtype))
            else:
                shape = projected[0].shape
                if any(tensor.shape != shape for tensor in projected):
                    raise ValueError(f"Order-{order} inverse Fourier contributions must share shape")
                output.append(torch.stack(projected, dim=0).sum(dim=0))
        return RealUpdate(output)


def _coerce_metadata(metadata: IrrepMetadata | str | Path | None) -> IrrepMetadata:
    if metadata is None:
        return load_default_irrep_metadata()
    if isinstance(metadata, IrrepMetadata):
        return metadata
    return IrrepMetadata.load(metadata)


def _fourier_block(
    tensor: torch.Tensor,
    partition: Partition,
    *,
    tuple_axis_start: int,
    metadata: IrrepMetadata,
) -> torch.Tensor:
    order = partition.order
    dim = irrep_dimension(partition)
    output = tensor.new_zeros(*tensor.shape, dim, dim)
    for permutation in all_permutations(order):
        matrix = metadata.representation_matrix(
            partition,
            permutation,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        permuted = permute_tuple_slots(tensor, permutation, axis_start=tuple_axis_start, order=order)
        output = output + permuted.unsqueeze(-1).unsqueeze(-1) * matrix
    return output


def _inverse_fourier_block(
    tensor: torch.Tensor,
    partition: Partition,
) -> torch.Tensor:
    order = partition.order
    dim = irrep_dimension(partition)
    trace = tensor.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return (dim / float(_factorial(order))) * trace


def _factorial(value: int) -> int:
    result = 1
    for item in range(2, value + 1):
        result *= item
    return result


__all__ = [
    "FourierTransform",
    "InverseFourierTransform",
]
