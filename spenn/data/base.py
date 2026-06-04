"""Base state and map contracts for SpechtMP scaffolds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any
from typing import Protocol, runtime_checkable

import torch
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
class ConcatenatedState(SpechtMPState):
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

        try:
            original_output = self(input)
            if not isinstance(original_output, SpechtMPState):
                raise TypeError("EquivariantMap outputs must implement SpechtMPState")
            transformed_output = self(input.permute(permutation))
            expected_output = original_output.permute(permutation)
            _assert_allclose(transformed_output, expected_output, atol=atol, rtol=rtol)
        except AssertionError:
            return False
        return True


def _assert_allclose(actual: Any, expected: Any, *, atol: float, rtol: float) -> None:
    """Assert equality or tensor closeness across state containers."""

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
                _assert_allclose(
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
            _assert_allclose(actual[key], expected[key], atol=atol, rtol=rtol)
        return
    if isinstance(actual, tuple) or isinstance(expected, tuple):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f"Tuple structure mismatch: {actual!r} != {expected!r}")
        for actual_item, expected_item in zip(actual, expected):
            _assert_allclose(actual_item, expected_item, atol=atol, rtol=rtol)
        return
    if (
        (isinstance(actual, Sequence) or isinstance(expected, Sequence))
        and not isinstance(actual, (str, bytes, bytearray))
        and not isinstance(expected, (str, bytes, bytearray))
    ):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f"Sequence structure mismatch: {actual!r} != {expected!r}")
        for actual_item, expected_item in zip(actual, expected):
            _assert_allclose(actual_item, expected_item, atol=atol, rtol=rtol)
        return
    if actual != expected:
        raise AssertionError(f"Values differ: {actual!r} != {expected!r}")


__all__ = ["ConcatenatedState", "EquivariantMap", "SpechtMPState"]
