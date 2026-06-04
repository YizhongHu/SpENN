"""Base state and map contracts for SpechtMP scaffolds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from torch import nn

from spenn.data.permutation import Permutation


@runtime_checkable
class SpechtMPState(Protocol):
    """Protocol for objects carrying a particle-permutation action."""

    def permute(self, permutation: Permutation) -> "SpechtMPState":
        """Return a copy transformed by `permutation`.

        Parameters
        ----------
        permutation : Permutation
            Permutation acting on particle-label axes.

        Returns
        -------
        SpechtMPState
            Permuted state.
        """

        ...


@dataclass(frozen=True)
class ConcatenatedState:
    """Bundle multiple SpechtMP states into one permutable state.

    Parameters
    ----------
    data : tuple of SpechtMPState
        Component states. The group action is applied componentwise.
    """

    data: tuple[SpechtMPState, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        data = tuple(self.data)
        for state in data:
            if not isinstance(state, SpechtMPState):
                raise TypeError("ConcatenatedState entries must implement SpechtMPState")
        object.__setattr__(self, "data", data)

    def __len__(self) -> int:
        """Return the number of component states."""

        return len(self.data)

    def __iter__(self):
        """Iterate over component states."""

        return iter(self.data)

    def __getitem__(self, index: int) -> SpechtMPState:
        """Return one component state."""

        return self.data[index]

    def permute(self, permutation: Permutation) -> "ConcatenatedState":
        """Return a state with every component permuted."""

        return ConcatenatedState(tuple(state.permute(permutation) for state in self.data))


class EquivariantMap(nn.Module):
    """Base class for maps that should commute with particle permutations."""

    def forward(self, input: SpechtMPState) -> SpechtMPState:
        """Apply the map.

        Parameters
        ----------
        input : SpechtMPState
            Input state. Multiple logical inputs should be bundled in a
            :class:`ConcatenatedState`.

        Returns
        -------
        SpechtMPState
            Output state.

        Raises
        ------
        NotImplementedError
            Always raised by the base class.
        """

        raise NotImplementedError

    def is_equivariant(
        self,
        input: SpechtMPState,
        permutation: Permutation,
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-5,
    ) -> bool:
        """Return whether the module passes one equivariance check."""

        from spenn.testing.equivariance import assert_equivariant

        try:
            assert_equivariant(self, input, permutation, atol=atol, rtol=rtol)
        except AssertionError:
            return False
        return True


__all__ = ["ConcatenatedState", "EquivariantMap", "SpechtMPState"]
