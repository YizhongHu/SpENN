"""Tests for generic equivariance testing helpers."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.equivariant_map import EquivariantMap
from spenn.data.permutation import Permutation
from spenn.data.real_features import RealFeature
from spenn.testing import assert_equivariant, assert_tree_allclose, permute_tree


def _feature() -> RealFeature:
    return RealFeature(
        [
            torch.tensor([[1.0, 2.0]]),
            torch.arange(1 * 2 * 3, dtype=torch.float64).reshape(1, 2, 3),
            torch.arange(1 * 2 * 3 * 3, dtype=torch.float64).reshape(1, 2, 3, 3),
        ]
    )


def test_permute_tree_handles_nested_structures() -> None:
    permutation = Permutation((1, 2, 0))
    tensor = torch.arange(1 * 1 * 3, dtype=torch.float64).reshape(1, 1, 3)
    value = {"feature": _feature(), "items": [tensor, (tensor,)]}

    permuted = permute_tree(value, permutation)

    assert_tree_allclose(permuted["feature"], value["feature"].permute(permutation))
    index = list(permutation.inverse().image)
    assert torch.equal(permuted["items"][0], tensor[:, :, index])
    assert torch.equal(permuted["items"][1][0], tensor[:, :, index])


def test_assert_equivariant_accepts_identity_module() -> None:
    assert_equivariant(nn.Identity(), _feature(), Permutation((2, 0, 1)))


class IdentityEquivariantMap(EquivariantMap):
    """Identity map for checking the base equivariance method."""

    def forward(self, input: object) -> object:
        """Return the input unchanged."""

        return input


def test_equivariant_map_is_equivariant_delegates_to_helper() -> None:
    module = IdentityEquivariantMap()

    assert module.is_equivariant(_feature(), Permutation((1, 2, 0)))
