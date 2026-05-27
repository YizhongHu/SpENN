"""Fixed M=2 branching maps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from spenn.data.feature_dict import BranchDict, MessageDict
from spenn.data.partitions import Par, Partition


class BranchMap(nn.Module):
    """Apply fixed M=2 branching maps from messages to branched tensors.

    Parameters
    ----------
    M : int, optional
        Maximum retained feature order. Only values up to ``2`` are accepted in
        this scaffold.
    M_virtual : int, optional
        Maximum virtual message order. Only values up to ``2`` are accepted in
        this scaffold.
    maps : mapping or None, optional
        Optional fixed map data reserved for the future implementation.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(self, M: int = 2, M_virtual: int = 2, maps: Mapping | None = None, **_: Any) -> None:
        super().__init__()
        if M > 2 or M_virtual > 2:
            raise ValueError("BranchMap scaffold only supports M <= 2 and M_virtual <= 2")
        self.M = M
        self.M_virtual = M_virtual
        self.maps = maps

    def forward(self, messages: MessageDict) -> BranchDict:
        """Return branched intermediate tensors from irrep-keyed messages.

        Parameters
        ----------
        messages : MessageDict
            Aggregated messages to branch back toward persistent feature
            blocks.

        Returns
        -------
        BranchDict
            Branched intermediate tensors produced by fixed branching maps.

        """

        branches = BranchDict()
        for source, target in _branch_routes():
            if messages.has(source):
                branches.set(target, source, self.branch_irrep(messages, source, target))
        return branches

    def branch_irrep(self, messages: MessageDict, source: Partition, target: Partition) -> torch.Tensor:
        """Branch one source message irrep into one target feature irrep.

        Parameters
        ----------
        messages : MessageDict
            Aggregated messages to branch.
        source : Partition
            Source message irrep partition.
        target : Partition
            Target feature irrep partition.

        Returns
        -------
        torch.Tensor
            Branched tensor block for the requested source and target irreps.

        Raises
        ------
        KeyError
            If the requested source message is missing.
        ValueError
            If the requested route is not supported by the hard-coded M=2
            branch formulas.
        """

        message = _scalar_message(messages, source)
        if source == target and source in {Par("H"), Par("S"), Par("A")}:
            return _identity_branch(message, source)
        if source in {Par("S"), Par("A")} and target == Par("H"):
            return _pair_to_node_branch(message, source)
        raise ValueError(f"Unsupported M=2 branch route {source.parts} -> {target.parts}")


def _branch_routes() -> tuple[tuple[Partition, Partition], ...]:
    return (
        (Par("H"), Par("H")),
        (Par("S"), Par("S")),
        (Par("A"), Par("A")),
        (Par("S"), Par("H")),
        (Par("A"), Par("H")),
    )


def _scalar_message(messages: MessageDict, source: Partition) -> torch.Tensor:
    tensor = messages.get(source)
    if tensor is None:
        raise KeyError(f"Missing source message for partition {source.parts}")
    if tuple(tensor.shape[-2:]) != (1, 1):
        raise ValueError(f"BranchMap only supports scalar-tailed M=2 messages, got tail {tuple(tensor.shape[-2:])}")
    message = tensor[..., 0, 0]
    assert message.ndim == source.order + 2
    return message


def _identity_branch(message: torch.Tensor, source: Partition) -> torch.Tensor:
    if source == Par("H"):
        batch_size, channels, n_electrons = message.shape
        output = message.new_zeros(batch_size, channels, 1, n_electrons, n_electrons, 1, 1)
        for i in range(n_electrons):
            output[:, :, 0, i, i, 0, 0] = message[:, :, i]
        assert output.shape == (batch_size, channels, 1, n_electrons, n_electrons, 1, 1)
        return output
    batch_size, channels, n_electrons, _ = message.shape
    output = message.new_zeros(batch_size, channels, 1, n_electrons, n_electrons, n_electrons, n_electrons, 1, 1)
    for i in range(n_electrons):
        for j in range(n_electrons):
            if i == j:
                continue
            output[:, :, 0, i, j, i, j, 0, 0] = message[:, :, i, j]
    assert output.shape == (batch_size, channels, 1, n_electrons, n_electrons, n_electrons, n_electrons, 1, 1)
    return output


def _pair_to_node_branch(message: torch.Tensor, source: Partition) -> torch.Tensor:
    batch_size, channels, n_electrons, _ = message.shape
    output = message.new_zeros(batch_size, channels, 1, n_electrons, n_electrons, n_electrons, 1, 1)
    for i in range(n_electrons):
        for j in range(n_electrons):
            for k in range(n_electrons):
                if j == k:
                    continue
                if i == j:
                    output[:, :, 0, i, j, k, 0, 0] = message[:, :, j, k]
                elif i == k:
                    sign = 1.0 if source == Par("S") else -1.0
                    output[:, :, 0, i, j, k, 0, 0] = sign * message[:, :, j, k]
    assert output.shape == (batch_size, channels, 1, n_electrons, n_electrons, n_electrons, 1, 1)
    return output


__all__ = ["BranchMap"]
