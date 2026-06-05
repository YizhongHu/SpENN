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
        Partition shorthand, such as ``(2, 1)``, ``"(2, 1)"``, ``"V"``, or
        ``"A3"``. A single integer is treated as a one-part partition.
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
        Partition-like value. Tuples/lists are interpreted as parts, strings
        may use compact forms such as ``"(2,1)"``, and an integer is treated as
        the one-part partition of that integer.
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


def normalize_partition(order: int, spec: _PartitionSpec) -> Partition:
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

    return as_partition(spec, order, name="Partition")


def normalize_partition_keys(supported: Iterable[Partition]) -> set[Partition]:
    """Validate and normalize partition keys.

    Parameters
    ----------
    supported : iterable of Partition
        Candidate partition keys.

    Returns
    -------
    set of Partition
        Checked partition key set.
    """

    normalized: set[Partition] = set()
    for partition in supported:
        normalized.add(as_partition(partition))
    return normalized


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


def _parts_from_spec(spec: _PartitionSpec) -> tuple[int, ...]:
    if isinstance(spec, Partition):
        return spec.parts
    if isinstance(spec, tuple):
        return tuple(_coerce_int(part, "partition part") for part in spec)
    if isinstance(spec, list):
        return tuple(_coerce_int(part, "partition part") for part in spec)
    if isinstance(spec, str):
        alias = _alias_parts(spec)
        if alias is not None:
            return alias
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


def _alias_parts(value: str) -> tuple[int, ...] | None:
    compact = value.strip().replace(" ", "").upper()
    fixed = {
        "H": (1,),
        "S": (2,),
        "A": (1, 1),
        "T": (3,),
        "V": (2, 1),
        "E": (1, 1, 1),
    }
    if compact in fixed:
        return fixed[compact]
    if len(compact) >= 2 and compact[0] in {"S", "A", "V"} and compact[1:].isdigit():
        order = int(compact[1:])
        if order <= 0:
            raise ValueError("Indexed partition aliases require positive order")
        if compact[0] == "S":
            return (order,)
        if compact[0] == "A":
            return (1,) * order
        if order < 2:
            raise ValueError("Standard representation alias Vn requires n >= 2")
        return (order - 1, 1)
    return None


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


Par = Partition


__all__ = [
    "Par",
    "Partition",
    "as_partition",
    "format_partition",
    "integer_partitions",
    "normalize_partition",
    "normalize_partition_keys",
    "partition_size",
    "transpose_partition",
    "validate_partition",
]
