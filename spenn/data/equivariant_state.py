"""Equivariant state protocol for SpENN data objects.

The active particle-permutation convention is
``(pi x)[i_1, ..., i_m] = x[pi^{-1} i_1, ..., pi^{-1} i_m]``. Concrete state
objects implement this convention in their ``permute`` method.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

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


__all__ = ["ConcatenatedState", "EquivariantState"]
