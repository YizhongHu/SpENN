"""Virtual-support path enumeration for equivariant mixing."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations


@dataclass(frozen=True)
class VirtualPath:
    """Describe one bilinear virtual-support mixing path.

    Parameters
    ----------
    support_order : int
        Virtual support order ``s``.
    target_order : int
        Output tuple order ``m``.
    left_order : int
        Left input tuple order ``m1``.
    right_order : int
        Right input tuple order ``m2``.
    target_injection, left_injection, right_injection : tuple of int
        Injective maps into the virtual support, represented as zero-based
        images.
    """

    support_order: int
    target_order: int
    left_order: int
    right_order: int
    target_injection: tuple[int, ...]
    left_injection: tuple[int, ...]
    right_injection: tuple[int, ...]

    @property
    def input_support(self) -> set[int]:
        """Return virtual labels covered by the two input injections."""

        return set(self.left_injection) | set(self.right_injection)

    def as_tuple(self) -> tuple[int, int, int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Return the mathematical tuple ``(s, m, m1, m2, tau, tau1, tau2)``."""

        return (
            self.support_order,
            self.target_order,
            self.left_order,
            self.right_order,
            self.target_injection,
            self.left_injection,
            self.right_injection,
        )


def validate_virtual_path(path: VirtualPath, *, max_order: int) -> None:
    """Validate one virtual-support path.

    Parameters
    ----------
    path : VirtualPath
        Candidate path.
    max_order : int
        Maximum allowed virtual support order.
    """

    if path.support_order > max_order:
        raise ValueError(f"Virtual support order {path.support_order} exceeds max_order {max_order}")
    if path.target_order > path.support_order:
        raise ValueError("target_order must be <= support_order")
    for name, order, injection in (
        ("target", path.target_order, path.target_injection),
        ("left", path.left_order, path.left_injection),
        ("right", path.right_order, path.right_injection),
    ):
        if len(injection) != order:
            raise ValueError(f"{name} injection length must match its order")
        if len(set(injection)) != len(injection):
            raise ValueError(f"{name} injection must be injective")
        if any(label < 0 or label >= path.support_order for label in injection):
            raise ValueError(f"{name} injection labels must land in the virtual support")
    if path.input_support != set(range(path.support_order)):
        raise ValueError("left and right injections must cover the virtual support")


def enumerate_virtual_paths(
    max_order: int,
    *,
    target_order: int | None = None,
    left_order: int | None = None,
    right_order: int | None = None,
) -> tuple[VirtualPath, ...]:
    """Enumerate valid bilinear virtual-support paths.

    Parameters
    ----------
    max_order : int
        Maximum virtual support order ``M``.
    target_order, left_order, right_order : int or None, optional
        Optional order filters. If omitted, all positive orders up to
        `max_order` are considered.

    Returns
    -------
    tuple of VirtualPath
        Valid paths satisfying ``s <= M`` and input-support coverage.
    """

    if max_order <= 0:
        raise ValueError(f"max_order must be positive, got {max_order}")
    target_orders = _orders(max_order, target_order, "target_order")
    left_orders = _orders(max_order, left_order, "left_order")
    right_orders = _orders(max_order, right_order, "right_order")
    paths: list[VirtualPath] = []
    for m in target_orders:
        for m1 in left_orders:
            for m2 in right_orders:
                min_support = max(m, m1, m2)
                for support_order in range(min_support, max_order + 1):
                    for tau in permutations(range(support_order), m):
                        for tau1 in permutations(range(support_order), m1):
                            for tau2 in permutations(range(support_order), m2):
                                path = VirtualPath(
                                    support_order=support_order,
                                    target_order=m,
                                    left_order=m1,
                                    right_order=m2,
                                    target_injection=tuple(tau),
                                    left_injection=tuple(tau1),
                                    right_injection=tuple(tau2),
                                )
                                if path.input_support == set(range(support_order)):
                                    paths.append(path)
    return tuple(paths)


def _orders(max_order: int, value: int | None, name: str) -> tuple[int, ...]:
    if value is None:
        return tuple(range(1, max_order + 1))
    if value <= 0 or value > max_order:
        raise ValueError(f"{name} must be in [1, {max_order}], got {value}")
    return (value,)


__all__ = ["VirtualPath", "enumerate_virtual_paths", "validate_virtual_path"]
