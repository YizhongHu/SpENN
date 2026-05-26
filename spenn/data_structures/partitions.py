"""Integer partition metadata and helper placeholders.

This module owns the lightweight representation of a partition label used by
feature containers and Specht representation code. Algorithmic helpers are
intentionally sketched as docstring-first placeholders until the representation
fixture work lands.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from operator import index
from typing import TypeAlias


@dataclass(frozen=True)
class Partition:
    """Immutable integer partition label.

    Parameters
    ----------
    order:
        Integer being partitioned.
    parts:
        Positive integer parts. Inputs are canonicalized to loosely decreasing
        order during initialization, so equality and hashing use the canonical
        tuple stored on the frozen dataclass.
    """

    order: int
    parts: tuple[int, ...]

    def __post_init__(self) -> None:
        order = _coerce_int(self.order, "order")
        if order < 0:
            raise ValueError("Partition order must be nonnegative")

        try:
            raw_parts = tuple(_coerce_int(part, "partition part") for part in self.parts)
        except TypeError as exc:
            raise TypeError("Partition parts must be an iterable of positive integers") from exc

        if any(part <= 0 for part in raw_parts):
            raise ValueError("Partition parts must be positive integers")

        canonical_parts = tuple(sorted(raw_parts, reverse=True))
        if sum(canonical_parts) != order:
            raise ValueError("Partition parts must sum to the partition order")

        object.__setattr__(self, "order", order)
        object.__setattr__(self, "parts", canonical_parts)


PartitionLike: TypeAlias = Partition | tuple[int, ...] | list[int] | str | int


def normalize_partition(order: int, spec: PartitionLike) -> Partition:
    """Normalize a partition specifier into a canonical :class:`Partition`.

    Parameters
    ----------
    order:
        Integer being partitioned.
    spec:
        Partition object, tuple/list of parts, compact string such as
        ``"(2,1)"``, or single integer part such as ``2``.

    Raises
    ------
    ValueError
        If `spec` is a partition of a different order.
    TypeError
        If `spec` uses an unsupported representation.
    """

    order = _coerce_int(order, "order")
    if isinstance(spec, Partition):
        if spec.order != order:
            raise ValueError(f"Partition order mismatch: expected {order}, got {spec.order}")
        return spec
    return Partition(order=order, parts=_parts_from_spec(spec))


def validate_partition(order: int, parts: Iterable[int]) -> Partition:
    """Return a canonical :class:`Partition` if `order` and `parts` are valid.

    Validation is delegated to :class:`Partition`, including descending
    canonicalization, positivity checks, and the sum-to-order invariant.
    """

    return Partition(order=order, parts=tuple(parts))


def partition_size(parts: Iterable[int]) -> int:
    """Return the integer partitioned by `parts`.

    This is a small formatting/metadata helper, not a validator. Use
    :func:`validate_partition` when the caller needs a canonical partition
    object.
    """

    return sum(_coerce_int(part, "partition part") for part in parts)


def format_partition(partition: Partition) -> str:
    """Format a partition in compact Specht-label style.

    Examples
    --------
    ``Partition(3, (2, 1))`` formats as ``"(2,1)"``.
    """

    return "(" + ",".join(str(part) for part in partition.parts) + ")"


def integer_partitions(order: int) -> tuple[Partition, ...]:
    """Enumerate all integer partitions of `order`.

    Placeholder for the future combinatorics implementation. The eventual
    result should be ordered consistently with the representation fixture
    generation code.
    """

    raise NotImplementedError("Integer partition enumeration is not implemented yet")


def transpose_partition(partition: Partition) -> Partition:
    """Return the conjugate Young-diagram partition.

    Placeholder for the future Young-diagram helper implementation.
    """

    raise NotImplementedError("Partition transposition is not implemented yet")


def _parts_from_spec(spec: PartitionLike) -> tuple[int, ...]:
    if isinstance(spec, tuple):
        return tuple(_coerce_int(part, "partition part") for part in spec)
    if isinstance(spec, list):
        return tuple(_coerce_int(part, "partition part") for part in spec)
    if isinstance(spec, str):
        stripped = spec.strip().strip("()")
        if not stripped:
            return tuple()
        return tuple(_parse_int_text(part.strip(), "partition part") for part in stripped.split(",") if part.strip())
    if isinstance(spec, int):
        return (_coerce_int(spec, "partition part"),)
    raise TypeError(f"Unsupported partition specifier: {spec!r}")


def _coerce_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _parse_int_text(value: str, name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise TypeError(f"{name} must be an integer") from exc


__all__ = [
    "Partition",
    "PartitionLike",
    "format_partition",
    "integer_partitions",
    "normalize_partition",
    "partition_size",
    "transpose_partition",
    "validate_partition",
]
