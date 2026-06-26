"""Learned aggregation over path-resolved Specht coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from operator import index

import torch
from torch import nn

from spenn.data.irrep import IrrepFeature, IrrepInteraction
from spenn.data.partition import Partition, as_partition
from spenn.data.equivariant_map import EquivariantMap


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
    channel_out_by_order : int, mapping, or None, optional
        Output channels for each tuple order. If an integer is supplied, every
        order uses that many channels. If a mapping is supplied, it is keyed by
        integer order. If ``None``, each partition preserves its input channel
        count, so all same-order input partitions must already share channels.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        *,
        channel_out_by_order: int | Mapping[int, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.channel_out_by_order = _normalize_channel_out_by_order(channel_out_by_order)
        self.weights = nn.ParameterDict()

    def forward_impl(self, x: IrrepInteraction) -> IrrepFeature:
        """Return learned path-aggregated irrep features."""

        if self.channel_out_by_order is None:
            self._validate_default_output_channels(x)
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
        beta_out = beta_in
        out_channels = self._out_channels(partition.order, in_channels)
        shape = (out_channels, beta_out, in_channels, path_count, beta_in)
        key = self.key(partition)
        if key not in self.weights:
            weight = torch.empty(shape, device=tensor.device, dtype=tensor.dtype)
            if weight.numel() > 0:
                nn.init.xavier_uniform_(weight)
            self.weights[key] = nn.Parameter(weight)
            return self.weights[key]
        weight = self.weights[key]
        if tuple(weight.shape) != shape:
            raise ValueError(
                f"PathAggregation weight for {partition.parts} has shape {tuple(weight.shape)}, "
                f"expected {shape}"
            )
        return weight.to(device=tensor.device, dtype=tensor.dtype)

    def _out_channels(self, order: int, in_channels: int) -> int:
        if self.channel_out_by_order is None:
            return in_channels
        if isinstance(self.channel_out_by_order, int):
            return self.channel_out_by_order
        if order not in self.channel_out_by_order:
            raise ValueError(f"PathAggregation channel_out_by_order is missing order {order}")
        return self.channel_out_by_order[order]

    def _validate_default_output_channels(self, x: IrrepInteraction) -> None:
        channels_by_order: dict[int, int] = {}
        for partition, tensor in x.items():
            channels = int(tensor.shape[1])
            previous = channels_by_order.setdefault(partition.order, channels)
            if previous != channels:
                raise ValueError(
                    "PathAggregation with channel_out_by_order=None requires same-order "
                    f"partitions to share input channels; order {partition.order} has "
                    f"both {previous} and {channels}"
                )


def _normalize_channel_out_by_order(
    value: int | Mapping[int, int] | None,
) -> int | dict[int, int] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        normalized = {}
        for raw_order, raw_channels in value.items():
            order = _nonnegative_int(raw_order, "channel_out_by_order key")
            channels = _nonnegative_int(raw_channels, f"channel_out_by_order[{order}]")
            normalized[order] = channels
        return normalized
    return _nonnegative_int(value, "channel_out_by_order")


def _nonnegative_int(value: object, name: str) -> int:
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be nonnegative, got {result}")
    return result


__all__ = ["PathAggregation"]
