"""Tuple-index helpers for SpENN data states.

This module owns generic ordered-tuple bookkeeping. State containers and neural
modules should use these helpers instead of defining local copies of tuple-grid,
masking, or particle-axis permutation logic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations, permutations, product

import torch

from spenn.data.permutation import Permutation


def all_subsets(n_items: int, order: int) -> list[tuple[int, ...]]:
    """Return unordered subsets of a fixed order.

    Parameters
    ----------
    n_items : int
        Number of available labels.
    order : int
        Subset size.

    Returns
    -------
    list of tuple of int
        Subsets in lexicographic order.
    """

    _validate_nonnegative(n_items, "n_items")
    _validate_nonnegative(order, "order")
    return [tuple(item) for item in combinations(range(n_items), order)]


def all_pairs(n_items: int) -> list[tuple[int, int]]:
    """Return all unordered two-label subsets."""

    return all_subsets(n_items, 2)


def all_triples(n_items: int) -> list[tuple[int, int, int]]:
    """Return all unordered three-label subsets."""

    return all_subsets(n_items, 3)


def all_ordered_tuples(n_items: int, order: int, *, distinct: bool = True) -> list[tuple[int, ...]]:
    """Return ordered tuples of particle labels.

    Parameters
    ----------
    n_items : int
        Number of available labels.
    order : int
        Tuple order.
    distinct : bool, optional
        If ``True``, entries in each tuple must be distinct. If ``False``,
        repeated labels are allowed.

    Returns
    -------
    list of tuple of int
        Ordered tuples in deterministic iteration order.
    """

    _validate_nonnegative(n_items, "n_items")
    _validate_nonnegative(order, "order")
    iterator: Iterable[tuple[int, ...]]
    iterator = permutations(range(n_items), order) if distinct else product(range(n_items), repeat=order)
    return [tuple(item) for item in iterator]


def tuple_grid(n_items: int, order: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Return a dense ordered tuple grid.

    Parameters
    ----------
    n_items : int
        Number of particle labels.
    order : int
        Tuple order.
    device : torch.device, str, or None, optional
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[n_items, ..., n_items, order]``. There are
        `order` tuple-index axes before the final coordinate axis.
    """

    _validate_nonnegative(n_items, "n_items")
    _validate_nonnegative(order, "order")
    if order == 0:
        return torch.empty((), dtype=torch.long, device=device)
    axes = torch.meshgrid(
        *[torch.arange(n_items, device=device, dtype=torch.long) for _ in range(order)],
        indexing="ij",
    )
    return torch.stack(axes, dim=-1)


def diagonal_mask(n_items: int, order: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Return a mask for tuple entries with repeated labels.

    Parameters
    ----------
    n_items : int
        Number of particle labels.
    order : int
        Tuple order.
    device : torch.device, str, or None, optional
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Boolean mask over tuple-index axes. ``True`` marks tuples containing a
        repeated particle label.
    """

    if order <= 1:
        return torch.zeros((n_items,) * order, dtype=torch.bool, device=device)
    grid = tuple_grid(n_items, order, device=device)
    mask = torch.zeros(grid.shape[:-1], dtype=torch.bool, device=device)
    for left in range(order):
        for right in range(left + 1, order):
            mask = mask | (grid[..., left] == grid[..., right])
    return mask


def no_repeated_particle_mask(
    n_items: int,
    order: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a mask for tuples with all labels distinct."""

    return ~diagonal_mask(n_items, order, device=device)


def permutation_index(permutation: Permutation, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Return index-select indices for the active permutation convention.

    Parameters
    ----------
    permutation : Permutation
        Active particle-label permutation.
    device : torch.device, str, or None, optional
        Device for the returned index tensor.

    Returns
    -------
    torch.Tensor
        Long tensor containing ``permutation.inverse().image``.
    """

    return torch.tensor(permutation.inverse().image, device=device, dtype=torch.long)


def permute_tuple_axes(
    tensor: torch.Tensor,
    permutation: Permutation,
    *,
    axis_start: int,
    order: int,
) -> torch.Tensor:
    """Permute consecutive tuple-index axes of a tensor.

    The active convention is
    ``(pi x)[i_1, ..., i_m] = x[pi^{-1} i_1, ..., pi^{-1} i_m]``.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor containing tuple-index axes.
    permutation : Permutation
        Particle-label permutation.
    axis_start : int
        Position of the first tuple-index axis.
    order : int
        Number of consecutive tuple-index axes.

    Returns
    -------
    torch.Tensor
        Tensor transformed by the active permutation.
    """

    if order == 0:
        return tensor.clone()
    if tensor.shape[axis_start] != len(permutation):
        raise ValueError(
            f"Permutation of size {len(permutation)} is incompatible with "
            f"tuple axis length {tensor.shape[axis_start]}"
        )
    index = permutation_index(permutation, device=tensor.device)
    output = tensor
    for axis in range(axis_start, axis_start + order):
        if output.shape[axis] != len(permutation):
            raise ValueError(
                "All tuple-index axes must have the same particle count, "
                f"got axis {axis} length {output.shape[axis]}"
            )
        output = output.index_select(axis, index)
    return output


def validate_tuple_tensor(
    tensor: torch.Tensor,
    *,
    order: int,
    prefix_ndim: int,
    suffix_ndim: int = 0,
    name: str = "tensor",
) -> None:
    """Validate tensor rank and square tuple-index axes.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to validate.
    order : int
        Tuple order.
    prefix_ndim : int
        Number of axes before the tuple-index axes.
    suffix_ndim : int, optional
        Number of axes after the tuple-index axes.
    name : str, optional
        Name used in error messages.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if order < 0:
        raise ValueError(f"order must be nonnegative, got {order}")
    expected_ndim = prefix_ndim + order + suffix_ndim
    if tensor.ndim != expected_ndim:
        raise ValueError(
            f"Expected {name} with {expected_ndim} dimensions, got shape {tuple(tensor.shape)}"
        )
    if order == 0:
        return
    n_particles = tensor.shape[prefix_ndim]
    for axis in range(prefix_ndim, prefix_ndim + order):
        if tensor.shape[axis] != n_particles:
            raise ValueError(
                f"Expected all tuple axes in {name} to have length {n_particles}, "
                f"got axis {axis} length {tensor.shape[axis]}"
            )


def common_particle_count(
    blocks: Mapping[int, torch.Tensor],
    *,
    tuple_axis_start: int,
) -> int | None:
    """Return the shared particle count across non-scalar blocks.

    Parameters
    ----------
    blocks : mapping
        Mapping from tuple order to tensor blocks.
    tuple_axis_start : int
        Position of the first tuple-index axis in each tensor.

    Returns
    -------
    int or None
        Shared particle count, or ``None`` when no positive-order block exists.
    """

    n_particles: int | None = None
    for order, tensor in blocks.items():
        if order == 0:
            continue
        current = int(tensor.shape[tuple_axis_start])
        if n_particles is None:
            n_particles = current
        elif current != n_particles:
            raise ValueError(
                "Tuple tensor blocks must share one particle count, "
                f"got {n_particles} and {current}"
            )
    return n_particles


def _validate_nonnegative(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")


__all__ = [
    "all_ordered_tuples",
    "all_pairs",
    "all_subsets",
    "all_triples",
    "common_particle_count",
    "diagonal_mask",
    "no_repeated_particle_mask",
    "permutation_index",
    "permute_tuple_axes",
    "tuple_grid",
    "validate_tuple_tensor",
]
