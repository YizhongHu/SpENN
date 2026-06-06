"""Tuple-index helpers for SpENN data states.

This module owns generic ordered-tuple bookkeeping. State containers and neural
modules should use these helpers instead of defining local copies of tuple-grid,
masking, or particle-axis permutation logic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import permutations, product

import torch

from spenn.data.permutation import Permutation


def ordered_tuples(n_items: int, order: int, *, distinct: bool = True) -> list[tuple[int, ...]]:
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


def ordered_tuple_tensor(
    n_items: int,
    order: int,
    *,
    distinct: bool = True,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ordered particle-label tuples as a tensor.

    Parameters
    ----------
    n_items : int
        Number of available labels.
    order : int
        Tuple order.
    distinct : bool, optional
        If ``True``, entries in each tuple must be distinct.
    device : torch.device, str, or None, optional
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Long tensor with shape ``[n_tuples, order]``.
    """

    tuples = ordered_tuples(n_items, order, distinct=distinct)
    return torch.tensor(tuples, dtype=torch.long, device=device).reshape(len(tuples), order)


def select_tuple(source: tuple[int, ...], positions: tuple[int, ...]) -> tuple[int, ...]:
    """Select entries from `source` at zero-based `positions`.

    Parameters
    ----------
    source : tuple of int
        Source tuple, usually a virtual support tuple ``K``.
    positions : tuple of int
        Positions to select from `source`.

    Returns
    -------
    tuple of int
        ``tuple(source[position] for position in positions)``.
    """

    return tuple(source[position] for position in positions)


def select_tuple_tensor(source: torch.Tensor, positions: tuple[int, ...]) -> torch.Tensor:
    """Select tuple columns from a tensor of ordered tuples.

    Parameters
    ----------
    source : torch.Tensor
        Long tensor whose final axis stores tuple entries.
    positions : tuple of int
        Zero-based tuple-entry positions to select.

    Returns
    -------
    torch.Tensor
        Tensor with the same leading shape and final axis ``len(positions)``.
    """

    if source.ndim == 0:
        raise ValueError("source must have a tuple-entry axis")
    if any(position < 0 or position >= source.shape[-1] for position in positions):
        raise ValueError(f"positions {positions} are incompatible with tuple length {source.shape[-1]}")
    index = torch.tensor(positions, dtype=torch.long, device=source.device)
    return source.index_select(source.ndim - 1, index)


def flatten_tuple_indices(indices: torch.Tensor, n_items: int) -> torch.Tensor:
    """Flatten ordered tuple indices using row-major tuple-axis order.

    Parameters
    ----------
    indices : torch.Tensor
        Long tensor whose final axis stores tuple entries.
    n_items : int
        Number of particle labels per tuple axis.

    Returns
    -------
    torch.Tensor
        Long tensor with shape ``indices.shape[:-1]`` containing flattened
        indices for a dense tensor with shape ``[n_items] * order``.
    """

    _validate_nonnegative(n_items, "n_items")
    if indices.ndim == 0:
        raise ValueError("indices must have a tuple-entry axis")
    if indices.shape[-1] == 0:
        return torch.zeros(indices.shape[:-1], dtype=torch.long, device=indices.device)
    if torch.any((indices < 0) | (indices >= n_items)):
        raise ValueError(f"tuple indices must be in range [0, {n_items})")
    order = int(indices.shape[-1])
    powers = torch.tensor(
        [n_items ** exponent for exponent in reversed(range(order))],
        dtype=torch.long,
        device=indices.device,
    )
    return (indices.to(dtype=torch.long) * powers).sum(dim=-1)


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


def tuple_particle_inputs(particle_vectors: torch.Tensor, order: int) -> torch.Tensor:
    """Return dense channel-last particle-vector tuples for one order.

    Parameters
    ----------
    particle_vectors : torch.Tensor
        Per-particle vectors with shape ``[batch, n_particles, channels]``.
    order : int
        Tuple order to construct.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[batch, n_particles, ..., n_particles,
        order * channels]``. There are `order` tuple-index axes before the
        final channel axis.
    """

    if particle_vectors.ndim != 3:
        raise ValueError(
            "particle vectors must have shape [batch, n_particles, channels], "
            f"got {tuple(particle_vectors.shape)}"
        )
    if order <= 0:
        raise ValueError(f"order must be positive, got {order}")
    batch_size, n_particles, channels = particle_vectors.shape
    tuple_shape = (n_particles,) * order
    pieces = []
    for slot in range(order):
        view_shape = [batch_size, *([1] * order), channels]
        view_shape[1 + slot] = n_particles
        pieces.append(particle_vectors.reshape(view_shape).expand(batch_size, *tuple_shape, channels))
    return torch.cat(pieces, dim=-1)


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


def permute_particle_axis(
    tensor: torch.Tensor,
    permutation: Permutation,
    *,
    axis: int,
) -> torch.Tensor:
    """Permute one particle-indexed axis of a tensor.

    The active convention is ``(pi x)[i] = x[pi^{-1} i]``.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor containing a particle-indexed axis.
    permutation : Permutation
        Particle-label permutation.
    axis : int
        Axis to permute. Negative axes are accepted.

    Returns
    -------
    torch.Tensor
        Tensor transformed by the active permutation.
    """

    normalized_axis = axis if axis >= 0 else tensor.ndim + axis
    if normalized_axis < 0 or normalized_axis >= tensor.ndim:
        raise ValueError(f"axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions")
    if tensor.shape[normalized_axis] != len(permutation):
        raise ValueError(
            f"Permutation of size {len(permutation)} is incompatible with "
            f"axis {axis} length {tensor.shape[normalized_axis]}"
        )
    return tensor.index_select(normalized_axis, permutation_index(permutation, device=tensor.device))


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


def permute_tuple_slots(
    tensor: torch.Tensor,
    permutation: Permutation,
    *,
    axis_start: int,
    order: int,
) -> torch.Tensor:
    """Permute tuple slots by reordering tuple-index axes.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor containing `order` adjacent tuple-index axes.
    permutation : Permutation
        Slot permutation. The returned tensor satisfies
        ``out[..., i_0, ..., i_m] = tensor[..., i_{pi(0)}, ..., i_{pi(m)}]``.
    axis_start : int
        Position of the first tuple-index axis.
    order : int
        Number of adjacent tuple-index axes to reorder.

    Returns
    -------
    torch.Tensor
        Tensor with tuple slots permuted.
    """

    if order < 0:
        raise ValueError(f"order must be nonnegative, got {order}")
    if len(permutation) != order:
        raise ValueError(f"Permutation of size {len(permutation)} cannot permute {order} tuple slots")
    if order <= 1:
        return tensor
    normalized_axis = axis_start if axis_start >= 0 else tensor.ndim + axis_start
    if normalized_axis < 0 or normalized_axis + order > tensor.ndim:
        raise ValueError(f"Tuple axes [{axis_start}, {axis_start + order}) exceed tensor rank {tensor.ndim}")
    axes = list(range(tensor.ndim))
    tuple_axes = axes[normalized_axis : normalized_axis + order]
    axes[normalized_axis : normalized_axis + order] = [
        tuple_axes[index] for index in permutation.inverse().image
    ]
    return tensor.permute(*axes)


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
    blocks: Mapping[object, torch.Tensor] | Sequence[torch.Tensor],
    *,
    tuple_axis_start: int,
    order_getter: Callable[[object], int] | None = None,
    name: str = "Tuple tensor blocks",
) -> int | None:
    """Return the shared particle count across non-scalar blocks.

    Parameters
    ----------
    blocks : mapping or sequence
        Tensor blocks keyed or indexed by tuple order.
    tuple_axis_start : int
        Position of the first tuple-index axis in each tensor.
    order_getter : callable or None, optional
        Function mapping each key to its tuple order. If ``None``, the key
        itself is interpreted as the order.
    name : str, optional
        Name used in error messages.

    Returns
    -------
    int or None
        Shared particle count, or ``None`` when no positive-order block exists.
    """

    n_particles: int | None = None
    items = blocks.items() if isinstance(blocks, Mapping) else enumerate(blocks)
    for key, tensor in items:
        order = int(key if order_getter is None else order_getter(key))
        if order == 0:
            continue
        current = int(tensor.shape[tuple_axis_start])
        if n_particles is None:
            n_particles = current
        elif current != n_particles:
            raise ValueError(
                f"{name} must share one particle count, "
                f"got {n_particles} and {current}"
            )
    return n_particles


def _validate_nonnegative(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")


__all__ = [
    "common_particle_count",
    "diagonal_mask",
    "flatten_tuple_indices",
    "no_repeated_particle_mask",
    "ordered_tuples",
    "ordered_tuple_tensor",
    "permutation_index",
    "permute_particle_axis",
    "permute_tuple_axes",
    "permute_tuple_slots",
    "select_tuple",
    "select_tuple_tensor",
    "tuple_grid",
    "tuple_particle_inputs",
    "validate_tuple_tensor",
]
