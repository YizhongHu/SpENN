"""Tests for generic equivariance testing helpers."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.base import EquivariantMap
from spenn.data.permutation import Permutation
from spenn.data.real_features import RealFeature
from spenn.testing import assert_equivariant


def _feature() -> RealFeature:
    return RealFeature(
        [
            torch.tensor([[1.0, 2.0]]),
            torch.arange(1 * 2 * 3, dtype=torch.float64).reshape(1, 2, 3),
            torch.arange(1 * 2 * 3 * 3, dtype=torch.float64).reshape(1, 2, 3, 3),
        ]
    )


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
