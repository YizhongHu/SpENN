"""Learned aggregation over path-resolved Specht coordinates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from operator import index

import torch
from torch import nn

from spenn.data.irrep import IrrepFeature, IrrepInteraction
from spenn.data.partition import Partition, as_partition, integer_partitions
from spenn.equivariance import EquivariantMap
from spenn.reps.irreps import irrep_dimension
from spenn.reps.paths import PathMetadata, VirtualPath, load_default_path_metadata


class PathAggregation(EquivariantMap):
    """Aggregate path-resolved irrep interactions into irrep features.

    The input contract is :class:`IrrepInteraction` with blocks of shape
    ``[batch, c_in, paths, indices..., alpha, beta_in]``. The output contract is
    :class:`IrrepFeature` with blocks of shape
    ``[batch, c_out, indices..., alpha, beta_out]``. For each partition,
    ``beta_out`` is the irrep dimension, matching the validation contract of
    :class:`IrrepFeature`.

    The learned weight convention is
    ``[c_out, beta_out, c_in, paths, beta_in]``. The operation is shared over
    batch, tuple-index, and alpha axes, so the aggregation may mix channels,
    paths, and beta coordinates but cannot mix particles or transforming alpha
    coordinates.

    Parameters
    ----------
    max_order : int
        Maximum tuple order to aggregate.
    channels : int or mapping
        Input channels per tuple order.
    channel_out_by_order : int, mapping, or None, optional
        Output channels for each tuple order. If an integer is supplied, every
        order uses that many channels. If a mapping is supplied, it is keyed by
        integer order. If ``None``, each partition preserves its input channel
        count, so all same-order input partitions must already share channels.
    max_virtual_order : int or None, optional
        Maximum virtual support order used when deriving path counts from
        metadata. Defaults to `max_order`.
    paths : PathMetadata, tuple of VirtualPath, or None, optional
        Path metadata used to derive path counts. If ``None``, checked-in path
        metadata for `output_embedding` is loaded.
    partitions : iterable of Partition or None, optional
        Partitions to initialize. If ``None``, all integer partitions through
        `max_order` are initialized.
    path_counts_by_order : mapping of int to int or None, optional
        Explicit path counts, mainly for tests or custom path families.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        *,
        max_order: int,
        channels: int | Mapping[int, int],
        channel_out_by_order: int | Mapping[int, int] | None = None,
        max_virtual_order: int | None = None,
        paths: PathMetadata | tuple[VirtualPath, ...] | None = None,
        output_embedding: str = "canonical",
        partitions: Iterable[Partition] | None = None,
        path_counts_by_order: Mapping[int, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_order = int(max_order)
        self.max_virtual_order = self.max_order if max_virtual_order is None else int(max_virtual_order)
        if self.max_order <= 0:
            raise ValueError(f"max_order must be positive, got {self.max_order}")
        if self.max_virtual_order <= 0:
            raise ValueError(f"max_virtual_order must be positive, got {self.max_virtual_order}")
        self.channels_by_order = _normalize_positive_channels(channels, max_order=self.max_order, name="channels")
        self.channel_out_by_order = _normalize_channel_out_by_order(
            self.channels_by_order if channel_out_by_order is None else channel_out_by_order,
            max_order=self.max_order,
        )
        if path_counts_by_order is None:
            self.path_counts_by_order = _path_counts_by_order(
                paths=paths,
                output_embedding=output_embedding,
                max_order=self.max_order,
                max_virtual_order=self.max_virtual_order,
            )
        else:
            self.path_counts_by_order = _normalize_nonnegative_channels(
                path_counts_by_order,
                max_order=self.max_order,
                name="path_counts_by_order",
            )
        self.partitions = (
            tuple(partitions)
            if partitions is not None
            else tuple(partition for order in range(1, self.max_order + 1) for partition in integer_partitions(order))
        )
        self.weights = nn.ParameterDict()
        self._initialize_weights()

    def forward_impl(self, x: IrrepInteraction) -> IrrepFeature:
        """Return learned path-aggregated irrep features."""

        return IrrepFeature({partition: self.aggregate_block(partition, tensor) for partition, tensor in x.items()})

    def aggregate_block(self, partition: Partition, tensor: torch.Tensor) -> torch.Tensor:
        """Aggregate one partition block with the learned path weights.

        Parameters
        ----------
        partition : Partition
            Irrep partition labeling the block.
        tensor : torch.Tensor
            Path-resolved block with shape
            ``[batch, c_in, paths, indices..., alpha, beta_in]``.

        Returns
        -------
        torch.Tensor
            Path-aggregated block with shape
            ``[batch, c_out, indices..., alpha, beta_out]``.
        """

        partition = as_partition(partition)
        if tensor.ndim < 5:
            raise ValueError(f"PathAggregation block for {partition.parts} must have at least 5 dimensions")
        weight = self._weight_for(partition, tensor)
        return torch.einsum("bcp...ad,oecpd->bo...ae", tensor, weight)

    def key(self, partition: Partition) -> str:
        """Return the stable parameter key for one partition."""

        return as_partition(partition).key

    def _weight_for(self, partition: Partition, tensor: torch.Tensor) -> torch.Tensor:
        in_channels = int(tensor.shape[1])
        path_count = int(tensor.shape[2])
        beta_in = int(tensor.shape[-1])
        expected_in = self.channels_by_order[partition.order]
        expected_paths = self.path_counts_by_order[partition.order]
        expected_beta = irrep_dimension(partition)
        if in_channels != expected_in:
            raise ValueError(f"PathAggregation input channels for {partition.parts} are {in_channels}, expected {expected_in}")
        if path_count != expected_paths:
            raise ValueError(f"PathAggregation path count for {partition.parts} is {path_count}, expected {expected_paths}")
        if beta_in != expected_beta or int(tensor.shape[-2]) != expected_beta:
            raise ValueError(f"PathAggregation irrep dimensions for {partition.parts} must both be {expected_beta}")
        beta_out = expected_beta
        out_channels = self._out_channels(partition.order)
        shape = (out_channels, beta_out, in_channels, path_count, beta_in)
        key = self.key(partition)
        if key not in self.weights:
            raise RuntimeError(f"Missing eager PathAggregation weight for partition {partition.parts}")
        weight = self.weights[key]
        if tuple(weight.shape) != shape:
            raise ValueError(
                f"PathAggregation weight for {partition.parts} has shape {tuple(weight.shape)}, "
                f"expected {shape}"
            )
        return weight

    def _out_channels(self, order: int) -> int:
        return self.channel_out_by_order[order]

    def _initialize_weights(self) -> None:
        for partition in self.partitions:
            partition = as_partition(partition)
            if partition.order < 1 or partition.order > self.max_order:
                raise ValueError(f"partition {partition.parts} is outside max_order={self.max_order}")
            key = self.key(partition)
            if key in self.weights:
                continue
            dim = irrep_dimension(partition)
            shape = (
                self.channel_out_by_order[partition.order],
                dim,
                self.channels_by_order[partition.order],
                self.path_counts_by_order[partition.order],
                dim,
            )
            weight = torch.empty(shape)
            if weight.numel() > 0:
                nn.init.xavier_uniform_(weight)
            self.weights[key] = nn.Parameter(weight)


def _normalize_channel_out_by_order(
    value: int | Mapping[int, int],
    *,
    max_order: int,
) -> dict[int, int]:
    return _normalize_nonnegative_channels(value, max_order=max_order, name="channel_out_by_order")


def _normalize_positive_channels(
    value: int | Mapping[int, int],
    *,
    max_order: int,
    name: str,
) -> dict[int, int]:
    channels = _normalize_nonnegative_channels(value, max_order=max_order, name=name)
    for order, count in channels.items():
        if count <= 0:
            raise ValueError(f"{name}[{order}] must be positive, got {count}")
    return channels


def _normalize_nonnegative_channels(
    value: int | Mapping[int, int],
    *,
    max_order: int,
    name: str,
) -> dict[int, int]:
    if isinstance(value, Mapping):
        normalized = {}
        for raw_order, raw_channels in value.items():
            order = _nonnegative_int(raw_order, f"{name} key")
            if order < 1 or order > max_order:
                raise ValueError(f"{name} contains order {order} outside [1, {max_order}]")
            channels = _nonnegative_int(raw_channels, f"{name}[{order}]")
            normalized[order] = channels
        missing = [order for order in range(1, max_order + 1) if order not in normalized]
        if missing:
            raise ValueError(f"{name} is missing orders {missing}")
        return dict(sorted(normalized.items()))
    channels = _nonnegative_int(value, name)
    return {order: channels for order in range(1, max_order + 1)}


def _path_counts_by_order(
    *,
    paths: PathMetadata | tuple[VirtualPath, ...] | None,
    output_embedding: str,
    max_order: int,
    max_virtual_order: int,
) -> dict[int, int]:
    if isinstance(paths, PathMetadata):
        all_paths = paths.all_paths()
    elif paths is None:
        metadata = load_default_path_metadata(output_embedding)
        if metadata.max_order < max_order or metadata.max_virtual_order < max_virtual_order:
            raise ValueError(
                "Saved path metadata only covers "
                f"max_order={metadata.max_order}, max_virtual_order={metadata.max_virtual_order}; "
                "pass explicit PathMetadata for larger generated path families"
            )
        all_paths = metadata.all_paths()
    else:
        all_paths = list(paths)
    return {
        order: sum(
            1
            for path in all_paths
            if path.m == order
            and path.s <= max_virtual_order
            and path.m <= max_order
            and path.m1 <= max_order
            and path.m2 <= max_order
        )
        for order in range(1, max_order + 1)
    }


def _nonnegative_int(value: object, name: str) -> int:
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be nonnegative, got {result}")
    return result


__all__ = ["PathAggregation"]
