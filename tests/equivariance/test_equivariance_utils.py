"""Tests for equivariance checks on SpechtMP maps."""

from __future__ import annotations

import torch

from spenn.data.base import EquivariantMap
from spenn.data.permutation import Permutation
from spenn.data.real_features import RealFeature


def _feature() -> RealFeature:
    return RealFeature(
        [
            torch.tensor([[1.0, 2.0]]),
            torch.arange(1 * 2 * 3, dtype=torch.float64).reshape(1, 2, 3),
            torch.arange(1 * 2 * 3 * 3, dtype=torch.float64).reshape(1, 2, 3, 3),
        ]
    )


class IdentityEquivariantMap(EquivariantMap):
    """Identity map for checking the base equivariance method."""

    def forward(self, input: object) -> object:
        """Return the input unchanged."""

        return input


class LabelWeightedMap(EquivariantMap):
    """Apply fixed label weights to make a non-equivariant map."""

    def forward(self, input: RealFeature) -> RealFeature:
        """Return features with label-dependent order-one scaling."""

        weights = torch.tensor([1.0, 2.0, 4.0], dtype=input[1].dtype, device=input[1].device).reshape(1, 1, 3)
        return RealFeature([input[0].clone(), input[1] * weights, input[2].clone()])


def test_equivariant_map_is_equivariant_accepts_identity_map() -> None:
    module = IdentityEquivariantMap()

    assert module.is_equivariant(_feature(), Permutation((1, 2, 0)))


def test_equivariant_map_is_equivariant_rejects_label_dependent_map() -> None:
    module = LabelWeightedMap()

    assert not module.is_equivariant(_feature(), Permutation((1, 2, 0)))
