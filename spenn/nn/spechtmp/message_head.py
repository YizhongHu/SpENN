"""Trainable message aggregation scaffold for SpechtMP."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from spenn.data.feature_dict import FeatureDict, MessageDict, TensorProductDict
from spenn.data.partitions import Par, Partition, as_partition
from spenn.nn.utils.activations import Activation


class MessageHead(nn.Module):
    """Aggregate tensor-product features into irrep-keyed messages.

    Parameters
    ----------
    M : int, optional
        Maximum retained feature order. Only values up to ``2`` are accepted in
        this scaffold.
    M_virtual : int, optional
        Maximum virtual tensor-product order. Only values up to ``2`` are
        accepted in this scaffold.
    channels : mapping or None, optional
        Channel specification by body order. For example, ``[0, 32, 32]``
        creates 32 output channels for order-1 and order-2 messages.
    activation : Activation or None, optional
        Optional irrep-aware activation module. If ``None``, messages remain
        linear after aggregation.
    include_linear : bool, optional
        Whether the future message head should include the learned linear term
        from persistent features.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        M: int = 2,
        M_virtual: int = 2,
        channels: object | None = None,
        activation: Activation | None = None,
        include_linear: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        if M > 2 or M_virtual > 2:
            raise ValueError("MessageHead scaffold only supports M <= 2 and M_virtual <= 2")
        self.M = M
        self.M_virtual = M_virtual
        self.channels = channels
        self.activation = activation
        self.include_linear = include_linear
        self._out_channels = _resolve_channels(channels)
        self.product_projections = nn.ModuleDict(
            {
                _product_key(target, left, right): _lazy_irrep_linear(target, self._out_channels[target])
                for target, left, right in _message_routes()
                if self._out_channels[target] > 0
            }
        )
        self.linear_projections = nn.ModuleDict(
            {
                _irrep_key(partition): _lazy_irrep_linear(partition, self._out_channels[partition])
                for partition in _message_partitions()
                if include_linear and self._out_channels[partition] > 0
            }
        )

    def forward(self, products: TensorProductDict, features: FeatureDict | None = None) -> MessageDict:
        """Return aggregated messages from tensor-product features.

        Parameters
        ----------
        products : TensorProductDict
            Fixed tensor-product features produced by a separate fusion map.
        features : FeatureDict or None, optional
            Persistent input features used by the optional learned linear term.

        Returns
        -------
        MessageDict
            Aggregated messages keyed by target irrep.

        """

        messages = self.tensor_product_messages(products)
        if self.include_linear and features is not None:
            messages = _add_messages(messages, self.linear_messages(features))
        return self.apply_irrep_activation(messages)

    def linear_messages(self, features: FeatureDict) -> MessageDict:
        """Return the learned linear contribution to messages.

        Parameters
        ----------
        features : FeatureDict
            Persistent feature blocks used by the linear term.

        Returns
        -------
        MessageDict
            Linear message contribution.

        """

        messages = MessageDict()
        for partition, tensor in features.flat_items():
            key = _irrep_key(partition)
            if key not in self.linear_projections:
                continue
            module = self.linear_projections[key]
            messages.set(partition, _project_feature(module, partition, tensor))
        return messages

    def tensor_product_messages(self, products: TensorProductDict) -> MessageDict:
        """Return the learned tensor-product contribution to messages.

        Parameters
        ----------
        products : TensorProductDict
            Fixed tensor-product features.

        Returns
        -------
        MessageDict
            Tensor-product message contribution.

        """

        messages = MessageDict()
        for target, left, right, tensor in products.flat_items():
            key = _product_key(target, left, right)
            if key not in self.product_projections:
                continue
            module = self.product_projections[key]
            contribution = _project_product(module, target, left, right, tensor)
            existing = messages.get(target)
            messages.set(target, contribution if existing is None else existing + contribution)
        return messages

    def apply_irrep_activation(self, messages: MessageDict) -> MessageDict:
        """Apply an irrep-aware activation to messages.

        Parameters
        ----------
        messages : MessageDict
            Messages to activate.

        Returns
        -------
        MessageDict
            Activated messages.

        """

        if self.activation is None:
            return messages
        activated = self.activation(messages)
        if isinstance(activated, MessageDict):
            return activated
        if isinstance(activated, FeatureDict):
            return MessageDict({partition: tensor for partition, tensor in activated.flat_items()})
        raise TypeError("MessageHead activation must return a MessageDict or FeatureDict")


def _message_partitions() -> tuple[Partition, ...]:
    return (Par("H"), Par("S"), Par("A"))


def _message_routes() -> tuple[tuple[Partition, Partition, Partition], ...]:
    return (
        (Par("H"), Par("H"), Par("H")),
        (Par("S"), Par("H"), Par("H")),
        (Par("A"), Par("H"), Par("H")),
        (Par("S"), Par("H"), Par("S")),
        (Par("A"), Par("H"), Par("S")),
        (Par("S"), Par("H"), Par("A")),
        (Par("A"), Par("H"), Par("A")),
        (Par("S"), Par("S"), Par("H")),
        (Par("A"), Par("S"), Par("H")),
        (Par("S"), Par("A"), Par("H")),
        (Par("A"), Par("A"), Par("H")),
        (Par("S"), Par("S"), Par("S")),
        (Par("A"), Par("S"), Par("A")),
        (Par("A"), Par("A"), Par("S")),
        (Par("S"), Par("A"), Par("A")),
    )


def _resolve_channels(channels: object | None) -> dict[Partition, int]:
    default = {Par("H"): 1, Par("S"): 1, Par("A"): 1}
    if channels is None:
        return default
    if isinstance(channels, Sequence) and not isinstance(channels, (str, bytes)):
        resolved = default.copy()
        if len(channels) > 1:
            resolved[Par("H")] = int(channels[1])
        if len(channels) > 2:
            resolved[Par("S")] = int(channels[2])
            resolved[Par("A")] = int(channels[2])
        return resolved
    if isinstance(channels, Mapping):
        resolved = default.copy()
        for key, value in channels.items():
            if isinstance(key, int):
                if key == 1:
                    resolved[Par("H")] = int(value)
                elif key == 2:
                    resolved[Par("S")] = int(value)
                    resolved[Par("A")] = int(value)
                continue
            if key in {"order1", "1"} and isinstance(value, Mapping):
                resolved[Par("H")] = int(next(iter(value.values())))
                continue
            if key in {"order2", "2"} and isinstance(value, Mapping):
                order2_values = [int(v) for v in value.values()]
                if order2_values:
                    resolved[Par("S")] = order2_values[0]
                    resolved[Par("A")] = order2_values[-1]
                continue
            partition = as_partition(key)
            if partition in resolved:
                resolved[partition] = int(value)
        return resolved
    raise TypeError("channels must be a sequence, mapping, or None")


def _project_product(
    module: nn.Module,
    target: Partition,
    left: Partition,
    right: Partition,
    tensor: torch.Tensor,
) -> torch.Tensor:
    if tuple(tensor.shape[-2:]) != (1, 1):
        raise ValueError("MessageHead only supports scalar-tailed M=2 tensor products")
    source_start = 3 + target.order
    source_stop = source_start + left.order + right.order
    reduced = tensor.sum(dim=tuple(range(source_start, source_stop)))
    values = reduced[..., 0, 0]
    permuted = values.permute(0, *range(3, 3 + target.order), 1, 2)
    mixed = module(permuted.reshape(*permuted.shape[:-2], permuted.shape[-2] * permuted.shape[-1]))
    output = mixed.movedim(-1, 1).unsqueeze(-1).unsqueeze(-1)
    assert output.ndim == target.order + 4
    assert output.shape[0] == tensor.shape[0]
    assert output.shape[2 : 2 + target.order] == tensor.shape[3 : 3 + target.order]
    assert output.shape[-2:] == (1, 1)
    return output


def _project_feature(module: nn.Module, partition: Partition, tensor: torch.Tensor) -> torch.Tensor:
    if tuple(tensor.shape[-2:]) != (1, 1):
        raise ValueError("MessageHead only supports scalar-tailed M=2 features")
    values = tensor[..., 0, 0]
    permuted = values.permute(0, *range(2, 2 + partition.order), 1)
    mixed = module(permuted)
    output = mixed.movedim(-1, 1).unsqueeze(-1).unsqueeze(-1)
    assert output.ndim == partition.order + 4
    assert output.shape[0] == tensor.shape[0]
    assert output.shape[2 : 2 + partition.order] == tensor.shape[2 : 2 + partition.order]
    assert output.shape[-2:] == (1, 1)
    return output


def _add_messages(left: MessageDict, right: MessageDict) -> MessageDict:
    output = left.clone()
    for partition, tensor in right.flat_items():
        existing = output.get(partition)
        output.set(partition, tensor if existing is None else existing + tensor)
    return output


def _lazy_irrep_linear(partition: Partition, out_channels: int) -> nn.LazyLinear:
    return nn.LazyLinear(out_channels, bias=partition != Par("A"))


def _product_key(target: Partition, left: Partition, right: Partition) -> str:
    return f"{_irrep_key(target)}<-{_irrep_key(left)}:{_irrep_key(right)}"


def _irrep_key(partition: Partition) -> str:
    return f"{partition.order}:{','.join(str(part) for part in partition.parts)}"


__all__ = ["MessageHead"]
