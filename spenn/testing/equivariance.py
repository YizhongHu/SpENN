"""Utilities for checking permutation equivariance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

import torch
from torch import nn

from spenn.data.permutation import Permutation


def _permute_tensor(tensor: torch.Tensor, permutation: Permutation) -> torch.Tensor:
    """Permute tensor axes after batch and channel axes."""

    if tensor.ndim <= 2:
        return tensor.clone()
    for axis in range(2, tensor.ndim):
        if tensor.shape[axis] != len(permutation):
            raise ValueError(
                f"Permutation of size {len(permutation)} is incompatible with "
                f"tensor shape {tuple(tensor.shape)} on axis {axis}"
            )
    index = torch.tensor(permutation.inverse().image, device=tensor.device, dtype=torch.long)
    output = tensor
    for axis in range(2, tensor.ndim):
        output = output.index_select(axis, index)
    return output


def permute_tree(value: Any, permutation: Permutation) -> Any:
    """Apply a permutation across a nested state tree.

    Parameters
    ----------
    value : object
        Tensor, state object, dataclass, sequence, mapping, or scalar to
        transform.
    permutation : Permutation
        Particle-label permutation.

    Returns
    -------
    object
        Permuted copy of `value`.
    """

    if isinstance(value, torch.Tensor):
        return _permute_tensor(value, permutation)
    if value is None:
        return None
    if hasattr(value, "permute") and callable(value.permute):
        return value.permute(permutation)
    if is_dataclass(value) and not isinstance(value, type):
        field_values = {
            field.name: permute_tree(getattr(value, field.name), permutation)
            for field in fields(value)
            if field.init
        }
        return type(value)(**field_values)
    if isinstance(value, Mapping):
        return type(value)((key, permute_tree(item, permutation)) for key, item in value.items())
    if isinstance(value, tuple):
        return type(value)(permute_tree(item, permutation) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return type(value)(permute_tree(item, permutation) for item in value)
    return value


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
    input: Any,
    permutation: Permutation,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-5,
) -> None:
    """Assert that a module commutes with a permutation.

    The checked contract is ``module(permute_tree(input, permutation))`` equals
    ``permute_tree(module(input), permutation)``.

    Parameters
    ----------
    module : torch.nn.Module
        Module to check.
    input : object
        Input state.
    permutation : Permutation
        Permutation used for the check.
    atol : float, optional
        Absolute tolerance passed to tensor comparisons.
    rtol : float, optional
        Relative tolerance passed to tensor comparisons.
    """

    original_output = module(input)
    transformed_input = permute_tree(input, permutation)
    transformed_output = module(transformed_input)
    expected_output = permute_tree(original_output, permutation)
    assert_tree_allclose(transformed_output, expected_output, atol=atol, rtol=rtol)
