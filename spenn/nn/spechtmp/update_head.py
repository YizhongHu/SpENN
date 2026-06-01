"""Trainable update aggregation scaffold for SpechtMP."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from spenn.data.feature_dict import BranchDict, FeatureDict
from spenn.data.partitions import Par, Partition, as_partition
from spenn.nn.utils.activations import Activation


class UpdateHead(nn.Module):
    """Mix branched tensors into feature updates.

    `UpdateHead` is the trainable ``y -> dx`` stage of SpechtMP. It consumes
    fixed branched intermediate tensors and applies learned channel and
    multiplicity-coordinate mixing with coupling tensor ``U`` to produce
    persistent feature updates.

    Parameters
    ----------
    M : int, optional
        Maximum retained feature order. Only values up to ``2`` are accepted in
        this scaffold.
    channels : mapping or None, optional
        Channel specification by body order. For example, ``[0, 32, 32]``
        creates 32 output channels for order-1 and order-2 feature updates.
    activation : Activation or None, optional
        Optional irrep-aware activation. The default ``None`` keeps update-head
        outputs linear; residual state updates are handled by `SpechtMPLayer`.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        M: int = 2,
        channels: object | None = None,
        activation: Activation | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if M > 2:
            raise ValueError("UpdateHead scaffold only supports M <= 2")
        self.M = M
        self.channels = channels
        self.activation = activation
        self._out_channels = _resolve_channels(channels)
        self.update_projections = nn.ModuleDict(
            {
                _branch_key(target, source): _lazy_irrep_linear(target, self._out_channels[target])
                for target, source in _update_routes()
                if self._out_channels[target] > 0
            }
        )

    def forward(self, branches: BranchDict, features: FeatureDict | None = None) -> FeatureDict:
        """Return feature updates from branched intermediate tensors.

        Parameters
        ----------
        branches : BranchDict
            Branched intermediate tensors produced by fixed branching.
        features : FeatureDict or None, optional
            Persistent input features reserved for future update-head variants.

        Returns
        -------
        FeatureDict
            Feature updates consumed by a final :class:`spenn.nn.utils.update.Update`
            rule.

        """

        return self.apply_irrep_activation(self.linear_updates(branches))

    def linear_updates(self, branches: BranchDict) -> FeatureDict:
        """Return learned linear updates from branched tensors.

        Parameters
        ----------
        branches : BranchDict
            Branched intermediate tensors produced by fixed branching.

        Returns
        -------
        FeatureDict
            Linear update contribution.

        """

        updates = FeatureDict()
        for target, source, tensor in branches.flat_items():
            key = _branch_key(target, source)
            if key not in self.update_projections:
                continue
            module = self.update_projections[key]
            contribution = _project_branch(module, target, source, tensor)
            existing = updates.get(target)
            updates.set(target, contribution if existing is None else existing + contribution)
        return updates

    def apply_irrep_activation(self, updates: FeatureDict) -> FeatureDict:
        """Apply an irrep-aware activation to feature updates.

        Parameters
        ----------
        updates : FeatureDict
            Feature updates to activate.

        Returns
        -------
        FeatureDict
            Activated feature updates.

        """

        if self.activation is None:
            return updates
        activated = self.activation(updates)
        if not isinstance(activated, FeatureDict):
            raise TypeError("UpdateHead activation must return a FeatureDict")
        return activated


def _update_routes() -> tuple[tuple[Partition, Partition], ...]:
    return (
        (Par("H"), Par("H")),
        (Par("H"), Par("S")),
        (Par("H"), Par("A")),
        (Par("S"), Par("S")),
        (Par("A"), Par("A")),
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


def _project_branch(module: nn.Module, target: Partition, source: Partition, tensor: torch.Tensor) -> torch.Tensor:
    if tuple(tensor.shape[-2:]) != (1, 1):
        raise ValueError("UpdateHead only supports scalar-tailed M=2 branches")
    source_start = 3 + target.order
    source_stop = source_start + source.order
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


def _lazy_irrep_linear(partition: Partition, out_channels: int) -> nn.LazyLinear:
    return nn.LazyLinear(out_channels, bias=partition != Par("A"))


def _branch_key(target: Partition, source: Partition) -> str:
    return f"{_irrep_key(target)}<-{_irrep_key(source)}"


def _irrep_key(partition: Partition) -> str:
    return f"{partition.order}:{','.join(str(part) for part in partition.parts)}"


__all__ = ["UpdateHead"]
