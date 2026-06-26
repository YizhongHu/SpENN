"""Equivariant state protocol for SpENN data objects.

The active particle-permutation convention is
``(pi x)[i_1, ..., i_m] = x[pi^{-1} i_1, ..., pi^{-1} i_m]``. Concrete state
objects implement this convention in their ``permute`` method.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

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


def permute_tree(obj: Any, permutation: Permutation) -> Any:
    """Apply a particle permutation to every equivariant object in a tree."""

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


def validate_tree(obj: Any) -> None:
    """Call ``validate`` on every validating object in a nested tree.

    Parameters
    ----------
    obj : object
        Tree containing mappings, sequences, and leaves that may expose a
        callable ``validate`` method.
    """

    validate = getattr(obj, "validate", None)
    if callable(validate):
        validate()
        return
    if isinstance(obj, Mapping):
        for value in obj.values():
            validate_tree(value)
        return
    if _is_sequence(obj):
        for value in obj:
            validate_tree(value)


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


__all__ = [
    "ConcatenatedState",
    "EquivariantState",
    "infer_particle_count",
    "permute_tree",
    "validate_tree",
]
