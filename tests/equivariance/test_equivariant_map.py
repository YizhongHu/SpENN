"""Tests for test-time equivariance helpers over EquivariantMap subclasses.

Runtime equivariance is no longer checked inside ``EquivariantMap.forward``
(see issue #19); the base class is a pure compute + passive tracer. Equivariance
is asserted here with the explicit test-time helpers in
``spenn.testing.equivariance``. Passive tracing behaviour is covered under
``tests/unit/equivariance/``.
"""

from __future__ import annotations

import pytest
import torch

from spenn.data.equivariant_state import validate_tree
from spenn.data.permutation import Permutation
from spenn.data.real import RealFeature, zero_block
from spenn.equivariance import EquivariantMap
from spenn.testing.equivariance import (
    assert_equivariant,
    assert_equivariant_all,
    equivariance_permutations,
)


def _feature() -> RealFeature:
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


class ValidatedLeaf:
    def __init__(self, calls: list[str], name: str, *, fail: bool = False) -> None:
        self.calls = calls
        self.name = name
        self.fail = fail

    def validate(self) -> "ValidatedLeaf":
        self.calls.append(self.name)
        if self.fail:
            raise ValueError(f"{self.name} failed validation")
        return self


def test_assert_equivariant_all_passes_equivariant_map() -> None:
    assert_equivariant_all(IdentityMap(), _feature())


def test_assert_equivariant_all_catches_non_equivariant_map() -> None:
    with pytest.raises(AssertionError):
        assert_equivariant_all(LabelWeightedMap(), _feature())


def test_assert_equivariant_helper_uses_forward_impl() -> None:
    assert_equivariant(IdentityMap(), _feature(), Permutation((1, 2, 0)), atol=0.0, rtol=0.0)


def test_assert_equivariant_all_owns_runtime_permutation_loop() -> None:
    assert_equivariant_all(IdentityMap(), _feature(), atol=0.0, rtol=0.0, max_full_size=3)


def test_small_runtime_checks_are_exhaustive() -> None:
    permutations = equivariance_permutations((_feature(),), max_full_size=3)

    assert len(permutations) == 6
    assert Permutation((2, 1, 0)) in permutations


def test_validate_tree_traverses_inputs_and_kwargs() -> None:
    calls: list[str] = []
    tree = ((ValidatedLeaf(calls, "arg"),), {"extra": ValidatedLeaf(calls, "kwarg")})

    validate_tree(tree)

    assert calls == ["arg", "kwarg"]


def test_validate_tree_propagates_leaf_errors() -> None:
    with pytest.raises(ValueError, match="failed validation"):
        validate_tree([ValidatedLeaf([], "arg", fail=True)])
