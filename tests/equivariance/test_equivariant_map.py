"""Tests for the pytest-only equivariance helpers over EquivariantMap toys.

Equivariance is asserted via ``tests.helpers.equivariance`` (typed ``.compare`` /
``apply_particle_permutation``), not the removed ``spenn.testing`` surface or any
generic tree walker.
"""

from __future__ import annotations

import pytest
import torch

from spenn.data.permutation import Permutation
from spenn.data.real import RealFeature, zero_block
from spenn.equivariance import EquivariantMap
from tests.helpers.equivariance import assert_equivariant, assert_equivariant_all


def _feature() -> RealFeature:
    # Last axis is the particle index (3 particles); channels = 2.
    return RealFeature(
        [
            zero_block(dtype=torch.float64),
            torch.arange(1 * 2 * 3, dtype=torch.float64).reshape(1, 2, 3),
            torch.arange(1 * 2 * 3 * 3, dtype=torch.float64).reshape(1, 2, 3, 3),
        ]
    )


class IdentityMap(EquivariantMap):
    def forward_impl(self, x: RealFeature) -> RealFeature:
        return x.clone()


class LabelWeightedMap(EquivariantMap):
    def forward_impl(self, x: RealFeature) -> RealFeature:
        weights = torch.tensor([1.0, 2.0, 4.0], dtype=x.blocks[1].dtype).reshape(1, 1, 3)
        return RealFeature([x.blocks[0].clone(), x.blocks[1] * weights, x.blocks[2].clone()])


def test_helper_passes_for_equivariant_map() -> None:
    assert_equivariant_all(IdentityMap(), _feature())


def test_helper_catches_non_equivariant_map() -> None:
    with pytest.raises(AssertionError):
        assert_equivariant_all(LabelWeightedMap(), _feature())


def test_single_permutation_helper_passes() -> None:
    assert_equivariant(IdentityMap(), _feature(), Permutation((1, 2, 0)))


def test_single_permutation_helper_catches_violation() -> None:
    with pytest.raises(AssertionError):
        assert_equivariant(LabelWeightedMap(), _feature(), Permutation((1, 2, 0)))
