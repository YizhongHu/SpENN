"""Utilities for checking permutation equivariance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

import torch
from torch import nn

from spenn.data.base import SpechtMPState
from spenn.data.permutation import Permutation


def assert_tree_allclose(actual: Any, expected: Any, *, atol: float = 1.0e-6, rtol: float = 1.0e-5) -> None:
    """Assert approximate equality across a nested state tree.

    Parameters
    ----------
    actual : object
        Actual tree.
    expected : object
        Expected tree.
    atol : float, optional
        Absolute tolerance passed to tensor comparisons.
    rtol : float, optional
        Relative tolerance passed to tensor comparisons.

    Raises
    ------
    AssertionError
        If any leaf differs.
    """

    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
            raise AssertionError(f"Tensor type mismatch: {type(actual)!r} != {type(expected)!r}")
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(f"None mismatch: {actual!r} != {expected!r}")
        return
    if is_dataclass(actual) or is_dataclass(expected):
        if type(actual) is not type(expected):
            raise AssertionError(f"Dataclass type mismatch: {type(actual)!r} != {type(expected)!r}")
        for field in fields(actual):
            if field.init:
                assert_tree_allclose(
                    getattr(actual, field.name),
                    getattr(expected, field.name),
                    atol=atol,
                    rtol=rtol,
                )
        return
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        if type(actual) is not type(expected):
            raise AssertionError(f"Mapping type mismatch: {type(actual)!r} != {type(expected)!r}")
        if actual.keys() != expected.keys():
            raise AssertionError(f"Mapping keys differ: {actual.keys()} != {expected.keys()}")
        for key in actual:
            assert_tree_allclose(actual[key], expected[key], atol=atol, rtol=rtol)
        return
    if isinstance(actual, tuple) or isinstance(expected, tuple):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f"Tuple structure mismatch: {actual!r} != {expected!r}")
        for actual_item, expected_item in zip(actual, expected):
            assert_tree_allclose(actual_item, expected_item, atol=atol, rtol=rtol)
        return
    sequence_types = (str, bytes, bytearray)
    if (
        (isinstance(actual, Sequence) or isinstance(expected, Sequence))
        and not isinstance(actual, sequence_types)
        and not isinstance(expected, sequence_types)
    ):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f"Sequence structure mismatch: {actual!r} != {expected!r}")
        for actual_item, expected_item in zip(actual, expected):
            assert_tree_allclose(actual_item, expected_item, atol=atol, rtol=rtol)
        return
    if actual != expected:
        raise AssertionError(f"Values differ: {actual!r} != {expected!r}")


def assert_equivariant(
    module: nn.Module,
    input: SpechtMPState,
    permutation: Permutation,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-5,
) -> None:
    """Assert that a module commutes with a permutation.

    The checked contract is ``module(input.permute(permutation))`` equals
    ``module(input).permute(permutation)``.

    Parameters
    ----------
    module : torch.nn.Module
        Module to check.
    input : SpechtMPState
        Input state.
    permutation : Permutation
        Permutation used for the check.
    atol : float, optional
        Absolute tolerance passed to tensor comparisons.
    rtol : float, optional
        Relative tolerance passed to tensor comparisons.
    """

    original_output = module(input)
    transformed_input = input.permute(permutation)
    transformed_output = module(transformed_input)
    if not isinstance(original_output, SpechtMPState):
        raise TypeError("Equivariant module outputs must implement SpechtMPState")
    expected_output = original_output.permute(permutation)
    assert_tree_allclose(transformed_output, expected_output, atol=atol, rtol=rtol)
