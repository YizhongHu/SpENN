"""Base module for maps with a permutation equivariance contract."""

from __future__ import annotations

from torch import nn

from spenn.data.permutation import Permutation


class EquivariantMap(nn.Module):
    """Base class for maps that should commute with particle permutations."""

    def forward(self, input: object) -> object:
        """Apply the map.

        Parameters
        ----------
        input : object
            Input state.

        Returns
        -------
        object
            Output state.

        Raises
        ------
        NotImplementedError
            Always raised by the base class.
        """

        raise NotImplementedError

    def is_equivariant(
        self,
        input: object,
        permutation: Permutation,
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-5,
    ) -> bool:
        """Return whether the module passes one equivariance check.

        Parameters
        ----------
        input : object
            Input state to test.
        permutation : Permutation
            Permutation to apply.
        atol : float, optional
            Absolute tolerance passed to tensor comparisons.
        rtol : float, optional
            Relative tolerance passed to tensor comparisons.

        Returns
        -------
        bool
            ``True`` if the check passes, otherwise ``False``.
        """

        from spenn.testing.equivariance import assert_equivariant

        try:
            assert_equivariant(self, input, permutation, atol=atol, rtol=rtol)
        except AssertionError:
            return False
        return True
