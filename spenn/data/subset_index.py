"""Subset and ordered-tuple index helpers."""

from __future__ import annotations

from itertools import combinations, permutations, product


def subset_key(subset: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Return a canonical tuple key for subset lookups.

    Parameters
    ----------
    subset : tuple of int or list of int
        Particle indices to canonicalize.

    Returns
    -------
    tuple of int
        Sorted tuple containing the same indices.
    """

    return tuple(sorted(subset))


def subset_complement(universe: tuple[int, ...] | list[int], subset: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Return the sorted complement of a subset.

    Parameters
    ----------
    universe : tuple of int or list of int
        Complete set of particle indices.
    subset : tuple of int or list of int
        Indices to remove from `universe`.

    Returns
    -------
    tuple of int
        Sorted indices in `universe` that are absent from `subset`.
    """

    universe_set = set(universe)
    subset_set = set(subset)
    return tuple(sorted(universe_set.difference(subset_set)))


def all_subsets(n: int, order: int) -> list[tuple[int, ...]]:
    """Return all canonical subsets of a fixed order.

    Parameters
    ----------
    n : int
        Number of indexed particles.
    order : int
        Subset cardinality.

    Returns
    -------
    list of tuple of int
        Subsets in lexicographic combination order.

    Raises
    ------
    ValueError
        If `order` is negative.
    """

    if order < 0:
        raise ValueError("order must be non-negative")
    return [tuple(subset) for subset in combinations(range(n), order)]


def all_ordered_tuples(n: int, order: int, distinct: bool = True) -> list[tuple[int, ...]]:
    """Return ordered particle tuples of a fixed order.

    Parameters
    ----------
    n : int
        Number of indexed particles.
    order : int
        Tuple length.
    distinct : bool, optional
        If ``True``, repeated particle indices are excluded. If ``False``,
        repeated indices are allowed.

    Returns
    -------
    list of tuple of int
        Ordered tuples in iterator order.

    Raises
    ------
    ValueError
        If `order` is negative.
    """

    if order < 0:
        raise ValueError("order must be non-negative")
    iterator = permutations(range(n), order) if distinct else product(range(n), repeat=order)
    return [tuple(index_tuple) for index_tuple in iterator]


def all_pairs(n: int) -> list[tuple[int, int]]:
    """Return all pair indices with ``i < j``.

    Parameters
    ----------
    n : int
        Number of indexed particles.

    Returns
    -------
    list of tuple of int
        Pair indices in lexicographic combination order.
    """

    return [tuple(pair) for pair in all_subsets(n, 2)]


def all_triples(n: int) -> list[tuple[int, int, int]]:
    """Return all triple indices with ``i < j < k``.

    Parameters
    ----------
    n : int
        Number of indexed particles.

    Returns
    -------
    list of tuple of int
        Triple indices in lexicographic combination order.
    """

    return [tuple(triple) for triple in all_subsets(n, 3)]
