"""Real-space permutation action utilities."""

from __future__ import annotations

from spenn.data.permutation import Permutation


def as_permutation(value: Permutation | tuple[int, ...] | list[int]) -> Permutation:
    """Convert a sequence-like value to :class:`Permutation`.

    Parameters
    ----------
    value : Permutation or sequence of int
        Permutation-like value.

    Returns
    -------
    Permutation
        Normalized permutation object.
    """

    return value if isinstance(value, Permutation) else Permutation(tuple(value))


__all__ = ["Permutation", "as_permutation"]
