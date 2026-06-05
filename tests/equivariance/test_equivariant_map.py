"""Tests for runtime equivariance checking on modules."""

from __future__ import annotations

import pytest
import torch

from spenn.data import Permutation, RealFeature, zero_block
from spenn.nn import EquivariantMap
from spenn.testing.equivariance import assert_equivariant, assert_equivariant_all, equivariance_permutations


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


class ValidationEchoMap(EquivariantMap):
    def forward_impl(self, x: ValidatedLeaf, *, extra: ValidatedLeaf) -> dict[str, list[ValidatedLeaf]]:
        return {"output": [x, extra]}


def test_runtime_checker_passes_equivariant_module() -> None:
    module = IdentityMap(equivariance_check=True, check_probability=1.0)

    out = module(_feature())

    assert isinstance(out, RealFeature)


def test_runtime_checker_catches_non_equivariant_module() -> None:
    module = LabelWeightedMap(equivariance_check=True, check_probability=1.0)

    with pytest.raises(AssertionError):
        module(_feature())


def test_assert_equivariant_helper_uses_forward_impl() -> None:
    assert_equivariant(IdentityMap(), _feature(), Permutation((1, 2, 0)), atol=0.0, rtol=0.0)


def test_assert_equivariant_all_owns_runtime_permutation_loop() -> None:
    assert_equivariant_all(IdentityMap(), _feature(), atol=0.0, rtol=0.0, max_full_size=3)


def test_small_runtime_checks_are_exhaustive() -> None:
    permutations = equivariance_permutations((_feature(),), max_full_size=3)

    assert len(permutations) == 6
    assert Permutation((2, 1, 0)) in permutations


def test_runtime_tensor_validation_traverses_inputs_kwargs_and_outputs() -> None:
    calls: list[str] = []
    module = ValidationEchoMap(tensor_validation_check=True)

    output = module(ValidatedLeaf(calls, "arg"), extra=ValidatedLeaf(calls, "kwarg"))

    assert isinstance(output, dict)
    assert calls == ["arg", "kwarg", "arg", "kwarg"]


def test_runtime_tensor_validation_rejects_bad_probability_and_propagates_errors() -> None:
    with pytest.raises(ValueError, match="validation_probability"):
        ValidationEchoMap(tensor_validation_check=True, validation_probability=2.0)

    module = ValidationEchoMap(tensor_validation_check=True)
    with pytest.raises(ValueError, match="failed validation"):
        module(ValidatedLeaf([], "arg", fail=True), extra=ValidatedLeaf([], "kwarg"))
