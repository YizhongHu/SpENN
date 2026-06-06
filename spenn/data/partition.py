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


@dataclass(frozen=True, init=False)
class Partition:
    """Immutable integer partition label.

    Parameters
    ----------
    spec : object, optional
        Partition specification, such as ``(2, 1)`` or ``"(2, 1)"``. A single
        integer is treated as a one-part partition.
    order : int, optional
        Integer being partitioned. When supplied with `parts` or a shorthand
        `spec`, it is checked against the inferred order.
    parts : iterable of int, optional
        Positive integer parts. Inputs are canonicalized to loosely decreasing
        order during initialization, so equality and hashing use the canonical
        tuple stored on the frozen dataclass.
    """

    order: int
    parts: tuple[int, ...]

    def __init__(
        self,
        *args: object,
        order: int | None = None,
        parts: Iterable[int] | None = None,
    ) -> None:
        if len(args) == 0:
            if parts is None:
                raise TypeError("Partition requires a spec or order and parts")
        elif len(args) == 1:
            if parts is not None:
                raise TypeError("Partition parts cannot be supplied with a shorthand spec")
            parts = _parts_from_spec(args[0])
        elif len(args) == 2:
            if order is not None or parts is not None:
                raise TypeError("Partition accepts either positional or keyword order and parts, not both")
            order = _coerce_int(args[0], "order")
            parts = _parts_from_spec(args[1])
        else:
            raise TypeError("Partition accepts at most two positional arguments")

        raw_parts = _validate_parts(parts)
        inferred_order = sum(raw_parts)
        if order is None:
            order = inferred_order
        order = _coerce_int(order, "order")
        if order < 0:
            raise ValueError("Partition order must be nonnegative")

        if any(part <= 0 for part in raw_parts):
            raise ValueError("Partition parts must be positive integers")

        canonical_parts = tuple(sorted(raw_parts, reverse=True))
        if sum(canonical_parts) != order:
            raise ValueError("Partition parts must sum to the partition order")

        object.__setattr__(self, "order", order)
        object.__setattr__(self, "parts", canonical_parts)

    @property
    def key(self) -> str:
        """Return a stable module-safe key for this partition.

        Returns
        -------
        str
            String key suitable for use in ``torch.nn.ModuleDict``.
        """

        return "p" + "_".join(str(part) for part in self.parts)

    def is_symmetric(self) -> bool:
        """Return whether this partition labels the symmetric irrep."""

        return self.parts == (self.order,)

    def is_antisymmetric(self) -> bool:
        """Return whether this partition labels the antisymmetric irrep."""

        return self.parts == (1,) * self.order


_PartitionSpec: TypeAlias = Partition | tuple[int, ...] | list[int] | str | int


def as_partition(spec: _PartitionSpec, order: int | None = None, *, name: str = "partition") -> Partition:
    """Convert a partition-like value to :class:`Partition`.

    Parameters
    ----------
    spec : Partition, tuple, list, str, or int
        Partition-like value. Tuples/lists are interpreted as parts, numeric
        strings may use compact forms such as ``"(2,1)"``, and an integer is
        treated as the one-part partition of that integer.
    order : int or None, optional
        Required partition order. If ``None``, the order is inferred from
        `spec`.
    name : str, optional
        Name used in error messages.

    Returns
    -------
    Partition
        Canonical partition object.

    Raises
    ------
    ValueError
        If `order` is provided and disagrees with `spec`.
    TypeError
        If `spec` uses an unsupported representation.
    """

    if isinstance(spec, Partition):
        partition = spec
    else:
        partition = Partition(spec)

    if order is None:
        return partition

    normalized_order = _coerce_int(order, "order")
    if partition.order != normalized_order:
        raise ValueError(f"{name} order mismatch: expected {normalized_order}, got {partition.order}")
    return partition


def integer_partitions(order: int) -> tuple[Partition, ...]:
    """Enumerate all integer partitions of `order`.

    Parameters
    ----------
    order : int
        Nonnegative integer to partition.

    Returns
    -------
    tuple of Partition
        Partitions in deterministic decreasing lexicographic order, e.g.
        ``(3,), (2, 1), (1, 1, 1)`` for order 3.
    """

    order = _coerce_int(order, "order")
    if order < 0:
        raise ValueError(f"order must be nonnegative, got {order}")
    if order == 0:
        return (Partition(order=0, parts=()),)

    def generate(remaining: int, max_part: int) -> tuple[tuple[int, ...], ...]:
        if remaining == 0:
            return ((),)
        partitions: list[tuple[int, ...]] = []
        for part in range(min(max_part, remaining), 0, -1):
            for suffix in generate(remaining - part, part):
                partitions.append((part, *suffix))
        return tuple(partitions)

    return tuple(Partition(parts=parts) for parts in generate(order, order))


def _parts_from_spec(spec: _PartitionSpec) -> tuple[int, ...]:
    if isinstance(spec, Partition):
        return spec.parts
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


def _validate_parts(parts: Iterable[int] | None) -> tuple[int, ...]:
    if parts is None:
        raise TypeError("Partition parts must be provided")
    try:
        return tuple(_coerce_int(part, "partition part") for part in parts)
    except TypeError as exc:
        raise TypeError("Partition parts must be an iterable of positive integers") from exc


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
    "as_partition",
    "integer_partitions",
]
