"""Equivariant-state contracts for SpENN data objects.

This module is intentionally narrow. ``EquivariantState`` is the typed contract
used as input/output of an equivariant map: semantic particle permutation
(`Permutable`) plus semantic comparison (`ComparableState`). It does not include
validation, health metrics, tensor-tree traversal, or pytest assertions --
runtime validation is a separate contract in :mod:`spenn.data.validation`.

The active particle-permutation convention is
``(pi x)[i_1, ..., i_m] = x[pi^{-1} i_1, ..., pi^{-1} i_m]``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, Self, runtime_checkable

import torch

from spenn.data.permutation import Permutable, Permutation

JsonScalar = int | float | bool | str | None


@runtime_checkable
class ComparableState(Protocol):
    """Typed value with semantic comparison to another value of the same type."""

    def compare(
        self,
        other: Self,
        *,
        atol: float,
        rtol: float,
    ) -> tuple[bool, Mapping[str, JsonScalar]]:
        """Return ``(is_close, metrics)`` versus another value of the same type."""

        ...


@runtime_checkable
class EquivariantState(Permutable, ComparableState, Protocol):
    """Typed value usable as input/output of an equivariant map.

    Includes only the operations equivariance checks need: semantic particle
    permutation (`Permutable`) and semantic comparison (`ComparableState`). It
    does not include validation, health metrics, tensor-tree traversal,
    particle-count inference, or pytest assertions.
    """


def apply_particle_permutation(value: Any, permutation: Permutation) -> Any:
    """Apply a particle permutation to one semantic, typed value.

    RED BANNER:
    Do not add a generic tree walker or any recursive container prober as a
    replacement for this function. Particle permutation and comparison are
    semantic typed-data actions. Values used in equivariance checks must expose
    explicit ``.permute(...)`` and ``.compare(...)`` contracts. Runtime
    validation belongs to separate typed validation contracts such as
    ``.validate()`` / ``.validity_metrics()`` (see :mod:`spenn.data.validation`),
    not to ``EquivariantState``.

    This dispatches on the value's own permutation contract, requiring a
    ``permute`` method (any `Permutable`); it never infers a representation
    action from arbitrary tensor shapes or container structure.

    Raises
    ------
    TypeError
        If `value` does not expose a callable ``permute``.
    """

    permute = getattr(value, "permute", None)
    if not callable(permute):
        raise TypeError(
            f"apply_particle_permutation: {type(value).__name__} is not particle-permutable "
            "(no callable .permute); runtime equivariance needs semantic typed values."
        )
    return permute(permutation)


def _compare_tensor_pair(x: torch.Tensor, y: torch.Tensor, *, atol: float, rtol: float) -> tuple[bool, float]:
    if x.shape != y.shape:
        return False, float("inf")
    if x.numel() == 0:
        return True, 0.0
    error = float((x - y).abs().max().item())
    return bool(torch.allclose(x, y, atol=atol, rtol=rtol)), error


def compare_tensor_blocks(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, dict[str, JsonScalar]]:
    """Compare two ordered tensor-block sequences; return ``(is_close, metrics)``."""

    if len(a) != len(b):
        return False, {"max_abs_error": float("inf")}
    close = True
    max_abs_error = 0.0
    for x, y in zip(a, b):
        pair_close, error = _compare_tensor_pair(x, y, atol=atol, rtol=rtol)
        if not math.isfinite(error):
            return False, {"max_abs_error": float("inf")}
        close = close and pair_close
        max_abs_error = max(max_abs_error, error)
    return close, {"max_abs_error": max_abs_error}


def compare_tensor_mapping(
    a: Mapping[Any, torch.Tensor],
    b: Mapping[Any, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, dict[str, JsonScalar]]:
    """Compare two keyed tensor mappings; return ``(is_close, metrics)``."""

    if set(a.keys()) != set(b.keys()):
        return False, {"max_abs_error": float("inf")}
    close = True
    max_abs_error = 0.0
    for key in a:
        pair_close, error = _compare_tensor_pair(a[key], b[key], atol=atol, rtol=rtol)
        if not math.isfinite(error):
            return False, {"max_abs_error": float("inf")}
        close = close and pair_close
        max_abs_error = max(max_abs_error, error)
    return close, {"max_abs_error": max_abs_error}


__all__ = [
    "ComparableState",
    "EquivariantState",
    "JsonScalar",
    "apply_particle_permutation",
    "compare_tensor_blocks",
    "compare_tensor_mapping",
]
