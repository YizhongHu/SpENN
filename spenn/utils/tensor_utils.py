"""Tensor helpers used across the phase 1 implementation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def resolve_dtype(dtype: Any | None) -> torch.dtype | None:
    """Resolve common dtype spellings to a torch dtype.

    Parameters
    ----------
    dtype : object or None
        Dtype specification, such as ``"float64"``, ``torch.float64``, or
        ``None``.

    Returns
    -------
    torch.dtype or None
        Resolved torch dtype, or ``None`` when `dtype` is ``None``.

    Raises
    ------
    TypeError
        If `dtype` cannot be resolved.
    """

    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        name = dtype.lower().replace("torch.", "")
        mapping = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float": torch.float32,
            "double": torch.float64,
            "int64": torch.int64,
            "long": torch.int64,
            "bool": torch.bool,
        }
        if name in mapping:
            return mapping[name]
    raise TypeError(f"Unsupported dtype specification: {dtype!r}")


def pairwise_displacements(positions: torch.Tensor) -> torch.Tensor:
    """Return pairwise displacement vectors ``r_i - r_j``.

    Parameters
    ----------
    positions : torch.Tensor
        Tensor of shape ``[batch, n_electrons, spatial_dim]``.

    Returns
    -------
    torch.Tensor
        Pairwise displacement tensor with shape
        ``[batch, n_electrons, n_electrons, spatial_dim]``.
    """

    return positions.unsqueeze(2) - positions.unsqueeze(1)


def pairwise_distances(positions: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return pairwise distances.

    Parameters
    ----------
    positions : torch.Tensor
        Tensor of shape ``[batch, n_electrons, spatial_dim]``.
    eps : float, optional
        Minimum distance value used for numerical safety.

    Returns
    -------
    torch.Tensor
        Pairwise distances with shape ``[batch, n_electrons, n_electrons, 1]``.
    """

    disp = pairwise_displacements(positions)
    dist = torch.linalg.norm(disp, dim=-1, keepdim=True)
    if eps:
        dist = dist.clamp_min(eps)
    return dist


def symmetrize_pair_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Enforce exact pair symmetry on pair axes.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor whose pair axes are axes 1 and 2.

    Returns
    -------
    torch.Tensor
        Symmetrized tensor.
    """

    return 0.5 * (tensor + tensor.transpose(1, 2))


def antisymmetrize_pair_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Enforce exact pair antisymmetry on pair axes.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor whose pair axes are axes 1 and 2.

    Returns
    -------
    torch.Tensor
        Antisymmetrized tensor.
    """

    return 0.5 * (tensor - tensor.transpose(1, 2))


def upper_triangle_indices(n: int) -> list[tuple[int, int]]:
    """Return upper-triangular pair indices.

    Parameters
    ----------
    n : int
        Matrix or particle-axis size.

    Returns
    -------
    list of tuple of int
        Pairs ``(i, j)`` with ``i < j``.
    """

    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def flatten_iterable(values: Iterable[Any]) -> list[Any]:
    """Convert a one-level iterable to a list.

    Parameters
    ----------
    values : iterable
        Values to materialize.

    Returns
    -------
    list
        Materialized values in iteration order.
    """

    return list(values)
