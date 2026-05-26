"""Subset and permutation helpers."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable


def normalize_subset(subset: Iterable[int]) -> tuple[int, ...]:
    """Return a canonical tuple representation of a subset.

    Parameters
    ----------
    subset : iterable of int
        Indices to canonicalize.

    Returns
    -------
    tuple of int
        Sorted tuple containing the same indices.
    """

    return tuple(sorted(subset))


def all_subsets(indices: Iterable[int], size: int | None = None) -> list[tuple[int, ...]]:
    """Enumerate subsets of ``indices``.

    If ``size`` is provided, only subsets of that cardinality are returned.

    Parameters
    ----------
    indices : iterable of int
        Indices to combine.
    size : int or None, optional
        Requested subset size. If ``None``, all subset sizes are returned.

    Returns
    -------
    list of tuple of int
        Subsets in increasing cardinality and combination order.
    """

    items = tuple(indices)
    if size is None:
        result = [()]
        for r in range(1, len(items) + 1):
            result.extend(tuple(combo) for combo in combinations(items, r))
        return result
    return [tuple(combo) for combo in combinations(items, size)]


def pair_indices(n: int) -> list[tuple[int, int]]:
    """Return all electron pairs ``(i, j)`` with ``i < j``.

    Parameters
    ----------
    n : int
        Number of indexed particles.

    Returns
    -------
    list of tuple of int
        Pair indices in lexicographic combination order.
    """

    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def triple_indices(n: int) -> list[tuple[int, int, int]]:
    """Return all electron triples ``(i, j, k)`` with ``i < j < k``.

    Parameters
    ----------
    n : int
        Number of indexed particles.

    Returns
    -------
    list of tuple of int
        Triple indices in lexicographic combination order.
    """

    return [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
