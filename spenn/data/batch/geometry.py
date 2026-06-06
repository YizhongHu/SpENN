"""Geometry helpers for electron batches."""

from __future__ import annotations

import torch

from spenn.data.batch.electron_batch import ElectronBatch


def pairwise_displacements(positions: torch.Tensor) -> torch.Tensor:
    """Return pairwise displacement vectors ``r_i - r_j``.

    Parameters
    ----------
    positions : torch.Tensor
        Tensor with shape ``[batch, n_electrons, spatial_dim]``.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[batch, n_electrons, n_electrons, spatial_dim]``.
    """

    _validate_positions(positions)
    return positions.unsqueeze(2) - positions.unsqueeze(1)


def pairwise_distances(positions: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """Return pairwise distances with a differentiable numerical floor.

    Parameters
    ----------
    positions : torch.Tensor
        Tensor with shape ``[batch, n_electrons, spatial_dim]``.
    eps : float, optional
        Positive distance floor. If nonzero, distances are
        ``sqrt(||r_i - r_j||^2 + eps^2)``.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[batch, n_electrons, n_electrons, 1]``.
    """

    displacement = pairwise_displacements(positions)
    squared = displacement.square().sum(dim=-1, keepdim=True)
    if eps:
        squared = squared + float(eps) ** 2
    return squared.sqrt()


def electron_nuclear_displacements(
    batch: ElectronBatch,
    nuclear_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return electron-nuclear displacement vectors ``r_i - R_A``.

    Parameters
    ----------
    batch : ElectronBatch
        Electron batch containing positions and nuclear coordinates. Nuclear
        coordinates may be shared across all samples or sampled with the same
        leading shape as the electron positions.
    nuclear_positions : torch.Tensor or None, optional
        Nuclear coordinates overriding `batch.nuclear_positions`, with shape
        ``[n_nuclei, spatial_dim]`` or ``[batch, n_nuclei, spatial_dim]``.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[batch, n_electrons, n_nuclei, spatial_dim]`` after
        flattening sample axes.

    Raises
    ------
    ValueError
        If nuclear positions are absent.
    """

    flat = batch.flatten_samples()
    nuclei = nuclear_positions if nuclear_positions is not None else flat.nuclear_positions
    if nuclei is None:
        raise ValueError("electron-nuclear displacements require nuclear positions")
    nuclei = nuclei.to(device=flat.device, dtype=flat.dtype)
    if nuclei.ndim == 2:
        if nuclei.shape[-1] != flat.spatial_dim:
            raise ValueError("nuclear positions must match batch spatial dimension")
        return flat.positions.unsqueeze(-2) - nuclei.reshape(1, 1, *nuclei.shape)
    if nuclei.ndim != 3:
        raise ValueError("nuclear positions must have shape [n_nuclei, dim] or [batch, n_nuclei, dim]")
    if nuclei.shape[0] != flat.batch_size or nuclei.shape[-1] != flat.spatial_dim:
        raise ValueError("batched nuclear positions must match batch size and spatial dimension")
    return flat.positions.unsqueeze(-2) - nuclei.unsqueeze(-3)


def electron_nuclear_distances(
    batch: ElectronBatch,
    eps: float = 1.0e-12,
    nuclear_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return electron-nuclear distances with a numerical floor.

    Parameters
    ----------
    batch : ElectronBatch
        Electron batch containing positions and nuclear coordinates.
    eps : float, optional
        Minimum returned distance.
    nuclear_positions : torch.Tensor or None, optional
        Nuclear coordinates overriding `batch.nuclear_positions`.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[batch, n_electrons, n_nuclei]`` after flattening
        sample axes.

    Raises
    ------
    ValueError
        If nuclear positions are absent.
    """

    return electron_nuclear_displacements(batch, nuclear_positions=nuclear_positions).norm(dim=-1).clamp_min(eps)


def nuclear_potential(batch: ElectronBatch, eps: float = 1.0e-12) -> torch.Tensor:
    """Return ``sum_A Z_A / |r_i - R_A|`` for each electron.

    Parameters
    ----------
    batch : ElectronBatch
        Electron batch containing positions, nuclear coordinates, and nuclear
        charges. Nuclear data may be shared or sampled with the electron
        positions.
    eps : float, optional
        Minimum electron-nuclear distance used in the denominator.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[batch, n_electrons]`` after flattening sample
        axes.

    Raises
    ------
    ValueError
        If nuclear positions or nuclear charges are absent.
    """

    if batch.nuclear_charges is None:
        raise ValueError("nuclear potential requires nuclear charges")
    flat = batch.flatten_samples()
    distances = electron_nuclear_distances(flat, eps=eps)
    charges = flat.nuclear_charges
    if charges is None:
        raise ValueError("nuclear potential requires nuclear charges")
    if charges.ndim == 1:
        charge_view = charges.reshape(1, 1, charges.shape[-1])
    else:
        charge_view = charges.unsqueeze(-2)
    return (charge_view / distances).sum(dim=-1)


def _validate_positions(positions: torch.Tensor) -> None:
    if positions.ndim != 3:
        raise ValueError(
            "positions must have shape [batch, n_electrons, spatial_dim], "
            f"got {tuple(positions.shape)}"
        )


__all__ = [
    "electron_nuclear_displacements",
    "electron_nuclear_distances",
    "nuclear_potential",
    "pairwise_displacements",
    "pairwise_distances",
]
