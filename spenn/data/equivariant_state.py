"""Equivariant state protocol for SpENN data objects.

The active particle-permutation convention is
``(pi x)[i_1, ..., i_m] = x[pi^{-1} i_1, ..., pi^{-1} i_m]``. Concrete state
objects implement this convention in their ``permute`` method.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from spenn.data.permutation import Permutation


@runtime_checkable
class EquivariantState(Protocol):
    """Protocol for objects carrying a particle-permutation action."""

    def permute(self, permutation: Permutation) -> "EquivariantState":
        """Return a copy transformed by a particle permutation.

        Parameters
        ----------
        permutation : Permutation
            Active particle-label permutation.

        Returns
        -------
        EquivariantState
            Permuted state object.
        """

        ...

    def compare(
        self,
        other: "EquivariantState",
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, float]:
        """Compare against another state, returning ``(is_close, max_abs_error)``.

        Equivariance checkers rely on this typed contract instead of inferring a
        comparison from arbitrary tensor-tree structure. A type or structural
        mismatch reports ``(False, inf)``.
        """

        ...


@dataclass(frozen=True)
class ConcatenatedState(EquivariantState):
    """Bundle multiple equivariant states into one permutable state.

    Parameters
    ----------
    data : tuple of EquivariantState
        Component states. The permutation action is applied componentwise.
    """

    data: tuple[EquivariantState, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        data = tuple(self.data)
        for state in data:
            if not isinstance(state, EquivariantState):
                raise TypeError("ConcatenatedState entries must implement EquivariantState")
        object.__setattr__(self, "data", data)

    def __len__(self) -> int:
        """Return the number of component states."""

        return len(self.data)

    def __iter__(self) -> Iterator[EquivariantState]:
        """Iterate over component states."""

        return iter(self.data)

    def __getitem__(self, index: int) -> EquivariantState:
        """Return one component state."""

        return self.data[index]

    def permute(self, permutation: Permutation) -> "ConcatenatedState":
        """Return a state with every component permuted."""

        return ConcatenatedState(tuple(state.permute(permutation) for state in self.data))

    def compare(
        self,
        other: "ConcatenatedState",
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, float]:
        """Compare componentwise; report ``(is_close, max_abs_error)``."""

        if type(self) is not type(other) or len(self.data) != len(other.data):
            return False, float("inf")
        close = True
        max_abs_error = 0.0
        for left, right in zip(self.data, other.data):
            entry_close, entry_error = left.compare(right, atol=atol, rtol=rtol)
            close = close and entry_close
            if not math.isfinite(entry_error):
                return False, float("inf")
            max_abs_error = max(max_abs_error, entry_error)
        return close, max_abs_error


def apply_particle_permutation(value: Any, permutation: Permutation) -> Any:
    """Apply a particle permutation to one semantic, typed value.

    RED BANNER:
    Do not add a generic tree walker or any recursive container prober as a
    replacement for this function. Particle permutation, validation, and
    comparison are semantic typed-data actions. Values used in equivariance
    checks must expose explicit ``.permute(...)``, ``.validate()``, and
    ``.compare(...)`` contracts.

    This dispatches on the value's own permutation contract, requiring a
    ``permute`` method (any `EquivariantState`); it never infers a representation
    action from arbitrary tensor shapes or container structure.

    Parameters
    ----------
    value : object
        A particle-permutable typed value exposing ``permute(permutation)``.
    permutation : Permutation
        Active particle-label permutation.

    Returns
    -------
    object
        The permuted value.

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


def infer_particle_count(obj: Any) -> int | None:
    """Infer a shared particle count from an input tree."""

    counts = _collect_particle_counts(obj)
    if not counts:
        return None
    first = counts[0]
    for count in counts[1:]:
        if count != first:
            raise ValueError(f"Equivariant inputs disagree on particle count: {counts}")
    return first


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
) -> tuple[bool, float]:
    """Compare two ordered tensor-block sequences; return ``(is_close, max_abs_error)``."""

    if len(a) != len(b):
        return False, float("inf")
    close = True
    max_abs_error = 0.0
    for x, y in zip(a, b):
        pair_close, error = _compare_tensor_pair(x, y, atol=atol, rtol=rtol)
        if not math.isfinite(error):
            return False, float("inf")
        close = close and pair_close
        max_abs_error = max(max_abs_error, error)
    return close, max_abs_error


def compare_tensor_mapping(
    a: Mapping[Any, torch.Tensor],
    b: Mapping[Any, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, float]:
    """Compare two keyed tensor mappings; return ``(is_close, max_abs_error)``."""

    if set(a.keys()) != set(b.keys()):
        return False, float("inf")
    close = True
    max_abs_error = 0.0
    for key in a:
        pair_close, error = _compare_tensor_pair(a[key], b[key], atol=atol, rtol=rtol)
        if not math.isfinite(error):
            return False, float("inf")
        close = close and pair_close
        max_abs_error = max(max_abs_error, error)
    return close, max_abs_error


__all__ = [
    "ConcatenatedState",
    "EquivariantState",
    "apply_particle_permutation",
    "compare_tensor_blocks",
    "compare_tensor_mapping",
    "infer_particle_count",
]
