"""Runtime equivariance assertions for particle-indexed state trees."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

import torch

from spenn.data.equivariant_state import infer_particle_count, permute_tree
from spenn.data.permutation import Permutation, adjacent_transpositions, all_permutations, reversal_permutation


def equivariance_permutation_schedule(
    size: int,
    *,
    max_full_size: int = 5,
) -> tuple[Permutation, ...]:
    """Return the runtime equivariance-check permutation schedule.

    Small systems are checked exhaustively. Larger systems check adjacent
    transpositions plus reversal, which keeps the runtime schedule
    deterministic while still exercising nonlocal label movement.

    Parameters
    ----------
    size : int
        Number of permuted particle labels.
    max_full_size : int, optional
        Maximum size checked by exhaustive enumeration.

    Returns
    -------
    tuple of Permutation
        Permutations used by runtime checks.
    """

    if size < 0:
        raise ValueError(f"Permutation size must be nonnegative, got {size}")
    if max_full_size < 0:
        raise ValueError(f"max_full_size must be nonnegative, got {max_full_size}")
    if size <= max_full_size:
        return all_permutations(size)
    images = tuple(permutation.image for permutation in adjacent_transpositions(size)) + (
        reversal_permutation(size).image,
    )
    return _unique_permutations(images)


def equivariance_permutations(
    obj: Any,
    *,
    max_full_size: int = 5,
) -> tuple[Permutation, ...]:
    """Return deterministic permutations for a runtime equivariance check.

    Parameters
    ----------
    obj : object
        Input tree from which a shared particle count can be inferred.
    max_full_size : int, optional
        Maximum particle count checked exhaustively.

    Returns
    -------
    tuple of Permutation
        Empty tuple when `obj` carries no particle-indexed state; otherwise
        the runtime permutation schedule for the inferred particle count.
    """

    n_particles = infer_particle_count(obj)
    if n_particles is None:
        return tuple()
    return equivariance_permutation_schedule(n_particles, max_full_size=max_full_size)


def assert_tree_allclose(a: Any, b: Any, *, atol: float, rtol: float) -> None:
    """Assert tensor closeness across nested state trees.

    Parameters
    ----------
    a, b : object
        Trees containing tensors, dataclasses, mappings, sequences, or scalar
        leaves.
    atol, rtol : float
        Absolute and relative tolerances passed to
        :func:`torch.testing.assert_close`.
    """

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise AssertionError(f"Tensor type mismatch: {type(a)!r} != {type(b)!r}")
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        return
    if a is None or b is None:
        if a is not b:
            raise AssertionError(f"None mismatch: {a!r} != {b!r}")
        return
    if is_dataclass(a) or is_dataclass(b):
        if type(a) is not type(b):
            raise AssertionError(f"Dataclass type mismatch: {type(a)!r} != {type(b)!r}")
        for field in fields(a):
            if field.init:
                assert_tree_allclose(getattr(a, field.name), getattr(b, field.name), atol=atol, rtol=rtol)
        return
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        if type(a) is not type(b):
            raise AssertionError(f"Mapping type mismatch: {type(a)!r} != {type(b)!r}")
        if a.keys() != b.keys():
            raise AssertionError(f"Mapping keys differ: {a.keys()} != {b.keys()}")
        for key in a:
            assert_tree_allclose(a[key], b[key], atol=atol, rtol=rtol)
        return
    if _is_sequence(a) or _is_sequence(b):
        if type(a) is not type(b) or len(a) != len(b):
            raise AssertionError(f"Sequence structure mismatch: {a!r} != {b!r}")
        for left, right in zip(a, b):
            assert_tree_allclose(left, right, atol=atol, rtol=rtol)
        return
    if a != b:
        raise AssertionError(f"Values differ: {a!r} != {b!r}")


def assert_equivariant(
    module: Any,
    inputs: Any,
    permutation: Permutation,
    *,
    kwargs: Mapping[str, Any] | None = None,
    original_output: Any | None = None,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-5,
) -> None:
    """Assert ``F(pi x) = pi F(x)`` for one module and permutation.

    Parameters
    ----------
    module : object
        Object exposing ``forward_impl`` or ``forward``.
    inputs : object or tuple
        Positional input tree.
    permutation : Permutation
        Particle-label permutation to check.
    kwargs : mapping or None, optional
        Keyword input tree.
    original_output : object or None, optional
        Optional cached ``F(x)`` value.
    atol, rtol : float, optional
        Tensor comparison tolerances.
    """

    args = inputs if isinstance(inputs, tuple) else (inputs,)
    kwargs = {} if kwargs is None else dict(kwargs)
    forward_impl = getattr(module, "forward_impl", None)
    if forward_impl is None:
        forward_impl = getattr(module, "forward")
    if original_output is None:
        original_output = forward_impl(*args, **kwargs)
    permuted_args = permute_tree(args, permutation)
    permuted_kwargs = permute_tree(kwargs, permutation)
    lhs = forward_impl(*permuted_args, **permuted_kwargs)
    rhs = permute_tree(original_output, permutation)
    assert_tree_allclose(lhs, rhs, atol=atol, rtol=rtol)


def assert_equivariant_all(
    module: Any,
    inputs: Any,
    *,
    kwargs: Mapping[str, Any] | None = None,
    original_output: Any | None = None,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-5,
    max_full_size: int = 5,
) -> None:
    """Assert equivariance over the runtime permutation schedule.

    Parameters
    ----------
    module : object
        Object exposing ``forward_impl`` or ``forward``.
    inputs : object or tuple
        Positional input tree.
    kwargs : mapping or None, optional
        Keyword input tree.
    original_output : object or None, optional
        Optional cached ``F(x)`` value.
    atol, rtol : float, optional
        Tensor comparison tolerances.
    max_full_size : int, optional
        Maximum particle count checked exhaustively.
    """

    args = inputs if isinstance(inputs, tuple) else (inputs,)
    kwargs = {} if kwargs is None else dict(kwargs)
    if original_output is None:
        forward_impl = getattr(module, "forward_impl", None)
        if forward_impl is None:
            forward_impl = getattr(module, "forward")
        original_output = forward_impl(*args, **kwargs)
    for permutation in equivariance_permutations((args, kwargs), max_full_size=max_full_size):
        assert_equivariant(
            module,
            args,
            permutation,
            kwargs=kwargs,
            original_output=original_output,
            atol=atol,
            rtol=rtol,
        )


def _unique_permutations(images: tuple[tuple[int, ...], ...]) -> tuple[Permutation, ...]:
    seen: set[tuple[int, ...]] = set()
    unique: list[Permutation] = []
    for image in images:
        if image not in seen:
            seen.add(image)
            unique.append(Permutation(image))
    return tuple(unique)


def _is_sequence(obj: Any) -> bool:
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))

__all__ = [
    "assert_equivariant",
    "assert_equivariant_all",
    "assert_tree_allclose",
    "equivariance_permutation_schedule",
    "equivariance_permutations",
    "infer_particle_count",
    "permute_tree",
]
