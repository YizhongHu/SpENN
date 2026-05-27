"""Fixed M=2 tensor-product maps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from spenn.data.feature_dict import FeatureDict, TensorProductDict
from spenn.data.partitions import Par, Partition


class FusionMap(nn.Module):
    """Compute fixed M=2 tensor-product features from feature blocks.

    Parameters
    ----------
    M : int, optional
        Maximum retained feature order. Only values up to ``2`` are accepted in
        this scaffold.
    M_virtual : int, optional
        Maximum virtual tensor-product order. Only values up to ``2`` are
        accepted in this scaffold.
    maps : mapping or None, optional
        Optional fixed map data reserved for the future implementation.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(self, M: int = 2, M_virtual: int = 2, maps: Mapping | None = None, **_: Any) -> None:
        super().__init__()
        if M > 2 or M_virtual > 2:
            raise ValueError("FusionMap scaffold only supports M <= 2 and M_virtual <= 2")
        self.M = M
        self.M_virtual = M_virtual
        self.maps = maps

    def forward(self, features: FeatureDict) -> TensorProductDict:
        """Return tensor-product features for an input feature dictionary.

        Parameters
        ----------
        features : FeatureDict
            Persistent feature blocks to fuse.

        Returns
        -------
        TensorProductDict
            Exact tensor-product feature blocks.

        """

        products = TensorProductDict()
        for target, left, right in _supported_products():
            if features.has(left) and features.has(right):
                products.set(target, left, right, self.fuse_pair(features, left, right, target))
        return products

    def fuse_pair(
        self,
        features: FeatureDict,
        left: Partition,
        right: Partition,
        target: Partition,
    ) -> torch.Tensor:
        """Fuse one pair of source irreps into a target irrep.

        Parameters
        ----------
        features : FeatureDict
            Persistent feature blocks to fuse.
        left : Partition
            Left source irrep partition.
        right : Partition
            Right source irrep partition.
        target : Partition
            Target irrep partition.

        Returns
        -------
        torch.Tensor
            Tensor-product block for the requested source and target irreps.

        Raises
        ------
        KeyError
            If a required source feature is missing.
        ValueError
            If the requested source and target irreps are unsupported by the
            hard-coded M=2 formulas.
        """

        left_tensor = _scalar_block(features, left)
        right_tensor = _scalar_block(features, right)
        if left == Par("H") and right == Par("H") and target == Par("H"):
            return _fuse_hh_to_h(left_tensor, right_tensor)
        if left == Par("H") and right == Par("H") and target in {Par("S"), Par("A")}:
            return _fuse_hh_to_pair(left_tensor, right_tensor, target=target)
        if left == Par("H") and right in {Par("S"), Par("A")} and target in {Par("S"), Par("A")}:
            return _fuse_h_pair_to_pair(left_tensor, right_tensor, target=target)
        if left in {Par("S"), Par("A")} and right == Par("H") and target in {Par("S"), Par("A")}:
            return _fuse_pair_h_to_pair(left_tensor, right_tensor, target=target)
        if left in {Par("S"), Par("A")} and right in {Par("S"), Par("A")} and target in {Par("S"), Par("A")}:
            if _pair_product_target(left, right) != target:
                raise ValueError(f"Unsupported M=2 fusion product {left.parts} x {right.parts} -> {target.parts}")
            return _fuse_pair_pair_to_pair(left_tensor, right_tensor, target=target)
        raise ValueError(f"Unsupported M=2 fusion product {left.parts} x {right.parts} -> {target.parts}")


def _supported_products() -> tuple[tuple[Partition, Partition, Partition], ...]:
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


def _scalar_block(features: FeatureDict, irrep: Partition) -> torch.Tensor:
    tensor = features.get(irrep)
    if tensor is None:
        raise KeyError(f"Missing source feature for partition {irrep.parts}")
    if tuple(tensor.shape[-2:]) != (1, 1):
        raise ValueError(f"FusionMap only supports scalar-tailed M=2 features, got tail {tuple(tensor.shape[-2:])}")
    block = tensor[..., 0, 0]
    assert block.ndim == irrep.order + 2
    return block


def _channel_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    batch_size = left.shape[0]
    left_channels = left.shape[1]
    right_channels = right.shape[1]
    left_tail = left.shape[2:]
    right_tail = right.shape[2:]
    left_view = left.reshape(batch_size, left_channels, 1, *left_tail, *([1] * len(right_tail)))
    right_view = right.reshape(batch_size, 1, right_channels, *([1] * len(left_tail)), *right_tail)
    output = (left_view * right_view).reshape(batch_size, left_channels * right_channels, *left_tail, *right_tail)
    assert output.shape == (batch_size, left_channels * right_channels, *left_tail, *right_tail)
    return output


def _fuse_hh_to_h(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    product = _channel_product(left, right)
    batch_size, channels, n_electrons, _ = product.shape
    output = product.new_zeros(batch_size, channels, 1, n_electrons, n_electrons, n_electrons, 1, 1)
    for i in range(n_electrons):
        output[:, :, 0, i, i, i, 0, 0] = product[:, :, i, i]
    assert output.shape == (batch_size, channels, 1, n_electrons, n_electrons, n_electrons, 1, 1)
    return output


def _fuse_hh_to_pair(left: torch.Tensor, right: torch.Tensor, *, target: Partition) -> torch.Tensor:
    product = _channel_product(left, right)
    batch_size, channels, n_electrons, _ = product.shape
    output = product.new_zeros(batch_size, channels, 1, n_electrons, n_electrons, n_electrons, n_electrons, 1, 1)
    sign = 1.0 if target == Par("S") else -1.0
    for i in range(n_electrons):
        for j in range(n_electrons):
            if i == j:
                continue
            output[:, :, 0, i, j, i, j, 0, 0] += 0.5 * product[:, :, i, j]
            output[:, :, 0, i, j, j, i, 0, 0] += 0.5 * sign * product[:, :, j, i]
    assert output.shape == (batch_size, channels, 1, n_electrons, n_electrons, n_electrons, n_electrons, 1, 1)
    return output


def _fuse_h_pair_to_pair(left: torch.Tensor, right: torch.Tensor, *, target: Partition) -> torch.Tensor:
    product = _channel_product(left, right)
    batch_size, channels, n_electrons, _, _ = product.shape
    output = product.new_zeros(batch_size, channels, 1, n_electrons, n_electrons, n_electrons, n_electrons, n_electrons, 1, 1)
    sign = 1.0 if target == Par("S") else -1.0
    for i in range(n_electrons):
        for j in range(n_electrons):
            if i == j:
                continue
            output[:, :, 0, i, j, i, i, j, 0, 0] += 0.5 * product[:, :, i, i, j]
            output[:, :, 0, i, j, j, j, i, 0, 0] += 0.5 * sign * product[:, :, j, j, i]
    assert output.shape == (batch_size, channels, 1, n_electrons, n_electrons, n_electrons, n_electrons, n_electrons, 1, 1)
    return output


def _fuse_pair_h_to_pair(left: torch.Tensor, right: torch.Tensor, *, target: Partition) -> torch.Tensor:
    product = _channel_product(left, right)
    batch_size, channels, n_electrons, _, _ = product.shape
    output = product.new_zeros(batch_size, channels, 1, n_electrons, n_electrons, n_electrons, n_electrons, n_electrons, 1, 1)
    sign = 1.0 if target == Par("S") else -1.0
    for i in range(n_electrons):
        for j in range(n_electrons):
            if i == j:
                continue
            output[:, :, 0, i, j, i, j, i, 0, 0] += 0.5 * product[:, :, i, j, i]
            output[:, :, 0, i, j, j, i, j, 0, 0] += 0.5 * sign * product[:, :, j, i, j]
    assert output.shape == (batch_size, channels, 1, n_electrons, n_electrons, n_electrons, n_electrons, n_electrons, 1, 1)
    return output


def _pair_product_target(left: Partition, right: Partition) -> Partition:
    if left == right:
        return Par("S")
    return Par("A")


def _fuse_pair_pair_to_pair(left: torch.Tensor, right: torch.Tensor, *, target: Partition) -> torch.Tensor:
    product = _channel_product(left, right)
    batch_size, channels, n_electrons, _, _, _ = product.shape
    output = product.new_zeros(
        batch_size,
        channels,
        1,
        n_electrons,
        n_electrons,
        n_electrons,
        n_electrons,
        n_electrons,
        n_electrons,
        1,
        1,
    )
    sign = 1.0 if target == Par("S") else -1.0
    for i in range(n_electrons):
        for j in range(n_electrons):
            if i == j:
                continue
            output[:, :, 0, i, j, i, j, i, j, 0, 0] += 0.5 * product[:, :, i, j, i, j]
            output[:, :, 0, i, j, j, i, j, i, 0, 0] += 0.5 * sign * product[:, :, j, i, j, i]
    assert output.shape == (
        batch_size,
        channels,
        1,
        n_electrons,
        n_electrons,
        n_electrons,
        n_electrons,
        n_electrons,
        n_electrons,
        1,
        1,
    )
    return output


__all__ = ["FusionMap"]
