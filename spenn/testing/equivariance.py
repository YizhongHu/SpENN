"""Runtime equivariance checking utilities.

These helpers intentionally favor reliability over speed. For small particle
counts they check every permutation, and for larger counts they check adjacent
transpositions plus reversal. Runtime module checks should call
``forward_impl`` rather than ``forward`` to avoid recursive checking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from itertools import permutations
from typing import Any

import torch

from spenn.data.permutation import Permutation


def permute_tree(obj: Any, permutation: Permutation) -> Any:
    """Apply a particle permutation to every equivariant object in a tree.

    Parameters
    ----------
    obj : object
        Object, dataclass, mapping, tuple, or list to transform.
    permutation : Permutation
        Active particle-label permutation.

    Returns
    -------
    object
        Object with all permutable leaves transformed. Non-equivariant leaves
        are returned unchanged.
    """

    permute = getattr(obj, "permute", None)
    if callable(permute):
        return permute(permutation)
    if isinstance(obj, Mapping):
        return type(obj)((key, permute_tree(value, permutation)) for key, value in obj.items())
    if isinstance(obj, tuple):
        return type(obj)(permute_tree(value, permutation) for value in obj)
    if isinstance(obj, list):
        return [permute_tree(value, permutation) for value in obj]
    return obj


def assert_tree_allclose(a: Any, b: Any, *, atol: float, rtol: float) -> None:
    """Assert tensor closeness across nested state trees.

    Parameters
    ----------
    a, b : object
        Objects to compare.
    atol : float
        Absolute tolerance for tensor comparisons.
    rtol : float
        Relative tolerance for tensor comparisons.
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
        Module exposing ``forward_impl`` or ``forward``.
    inputs : object
        Positional input object or tuple of positional inputs.
    permutation : Permutation
        Particle-label permutation used for the check.
    kwargs : mapping or None, optional
        Keyword inputs for the module.
    original_output : object or None, optional
        Precomputed unpermuted output. If ``None``, it is computed with
        ``forward_impl``.
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

    This function owns the loop over all permutations used by runtime checks.
    Small systems are checked exhaustively. Larger systems use the deterministic
    generator-style schedule returned by :func:`equivariance_permutations`.
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


def infer_particle_count(obj: Any) -> int | None:
    """Infer a shared particle count from an input tree.

    Parameters
    ----------
    obj : object
        Input tree.

    Returns
    -------
    int or None
        Inferred particle count, or ``None`` if no equivariant particle axis is
        discoverable.
    """

    counts = _collect_particle_counts(obj)
    if not counts:
        return None
    first = counts[0]
    for count in counts[1:]:
        if count != first:
            raise ValueError(f"Equivariant inputs disagree on particle count: {counts}")
    return first


def equivariance_permutations(
    obj: Any,
    *,
    max_full_size: int = 5,
) -> tuple[Permutation, ...]:
    """Return deterministic permutations for a runtime equivariance check.

    Parameters
    ----------
    obj : object
        Input tree used to infer the particle count.
    max_full_size : int, optional
        Check every permutation up to this particle count. Larger systems use
        adjacent transpositions plus reversal.

    Returns
    -------
    tuple of Permutation
        Permutations to check.
    """

    n_particles = infer_particle_count(obj)
    if n_particles is None:
        return tuple()
    if n_particles < 0:
        raise ValueError(f"Particle count must be nonnegative, got {n_particles}")
    if n_particles <= max_full_size:
        return tuple(Permutation(tuple(image)) for image in permutations(range(n_particles)))
    images: list[tuple[int, ...]] = []
    base = list(range(n_particles))
    for idx in range(n_particles - 1):
        image = base.copy()
        image[idx], image[idx + 1] = image[idx + 1], image[idx]
        images.append(tuple(image))
    images.append(tuple(reversed(base)))
    seen: set[tuple[int, ...]] = set()
    unique = []
    for image in images:
        if image not in seen:
            seen.add(image)
            unique.append(Permutation(image))
    return tuple(unique)


def _collect_particle_counts(obj: Any) -> list[int]:
    if obj is None:
        return []
    n_particles = getattr(obj, "n_particles", None)
    if n_particles is not None:
        return [int(n_particles)]
    n_electrons = getattr(obj, "n_electrons", None)
    if n_electrons is not None:
        return [int(n_electrons)]
    if isinstance(obj, Mapping):
        counts: list[int] = []
        for value in obj.values():
            counts.extend(_collect_particle_counts(value))
        return counts
    if _is_sequence(obj):
        counts = []
        for value in obj:
            counts.extend(_collect_particle_counts(value))
        return counts
    return []


def _is_sequence(obj: Any) -> bool:
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))


__all__ = [
    "assert_equivariant",
    "assert_equivariant_all",
    "assert_tree_allclose",
    "equivariance_permutations",
    "infer_particle_count",
    "permute_tree",
]
