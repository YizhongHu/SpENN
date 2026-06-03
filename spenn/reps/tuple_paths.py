"""Ordered-tuple path helpers for real-space Specht maps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TuplePath:
    """Describe one ordered-tuple route through a real-space map.

    Parameters
    ----------
    target : tuple of int
        Ordered target tuple.
    sources : tuple of tuple of int
        Ordered source tuples participating in the route.
    """

    target: tuple[int, ...]
    sources: tuple[tuple[int, ...], ...]


def tuple_union(*tuples: Sequence[int]) -> tuple[int, ...]:
    """Return labels from input tuples in first-seen order.

    Parameters
    ----------
    *tuples : sequence of int
        Ordered tuples whose labels should be combined.

    Returns
    -------
    tuple of int
        Label union preserving first occurrence.
    """

    output: list[int] = []
    seen: set[int] = set()
    for item in tuples:
        for label in item:
            normalized = int(label)
            if normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
    return tuple(output)


def restrict_ordered_tuple(target: Sequence[int], labels: Sequence[int]) -> tuple[int, ...]:
    """Restrict an ordered target tuple to a label set.

    Parameters
    ----------
    target : sequence of int
        Ordered target tuple.
    labels : sequence of int
        Labels to retain.

    Returns
    -------
    tuple of int
        Entries of `target` whose labels appear in `labels`, preserving target
        order.
    """

    label_set = {int(label) for label in labels}
    return tuple(int(label) for label in target if int(label) in label_set)


def enumerate_tuple_paths(
    n_electrons: int,
    target_order: int,
    source_orders: Sequence[int],
) -> tuple[TuplePath, ...]:
    """Enumerate ordered-tuple map paths.

    Parameters
    ----------
    n_electrons : int
        Number of electron labels.
    target_order : int
        Size of each target tuple.
    source_orders : sequence of int
        Sizes of source tuples in each path.

    Returns
    -------
    tuple of TuplePath
        Ordered-tuple paths for the requested map.

    Raises
    ------
    NotImplementedError
        Always raised until the generic tuple-path generator lands.
    """

    raise NotImplementedError("enumerate_tuple_paths is not implemented yet")


__all__ = ["TuplePath", "enumerate_tuple_paths", "restrict_ordered_tuple", "tuple_union"]
