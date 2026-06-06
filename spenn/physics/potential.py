"""Potential-energy terms."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, pairwise_distances
from spenn.physics.systems import ElectronicSystem


def harmonic_trap_potential(positions: torch.Tensor, omega: float = 1.0) -> torch.Tensor:
    """Return harmonic confinement energy for each configuration.

    Parameters
    ----------
    positions : torch.Tensor
        Electron coordinates with shape ``[batch, n_electrons, spatial_dim]``.
    omega : float, optional
        Harmonic trap frequency.

    Returns
    -------
    torch.Tensor
        Harmonic potential values with shape ``[batch]``.
    """

    if positions.ndim != 3:
        raise ValueError("positions must have shape [batch, n_electrons, spatial_dim]")
    output = 0.5 * (omega**2) * positions.square().sum(dim=(1, 2))
    assert output.shape == (positions.shape[0],)
    return output


def electron_electron_repulsion(positions: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return Coulomb repulsion summed over unique electron pairs.

    Parameters
    ----------
    positions : torch.Tensor
        Electron coordinates with shape ``[batch, n_electrons, spatial_dim]``.
    eps : float, optional
        Minimum pair distance used for numerical safety.

    Returns
    -------
    torch.Tensor
        Electron-electron repulsion values with shape ``[batch]``.
    """

    if positions.ndim != 3:
        raise ValueError("positions must have shape [batch, n_electrons, spatial_dim]")
    distances = pairwise_distances(positions, eps=eps).squeeze(-1)
    assert distances.shape == (positions.shape[0], positions.shape[1], positions.shape[1])
    tri = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    output = distances.reciprocal().masked_fill(~tri, 0.0).sum(dim=(1, 2))
    assert output.shape == (positions.shape[0],)
    return output


def electron_nuclear_attraction(
    positions: torch.Tensor,
    nuclear_positions: torch.Tensor,
    nuclear_charges: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return electron-nuclear attraction for batched positions.

    Parameters
    ----------
    positions : torch.Tensor
        Electron coordinates with shape ``[batch, n_electrons, spatial_dim]``.
    nuclear_positions : torch.Tensor
        Nuclear coordinates with shape ``[n_nuclei, spatial_dim]`` or
        ``[batch, n_nuclei, spatial_dim]``.
    nuclear_charges : torch.Tensor
        Nuclear charges with shape ``[n_nuclei]`` or ``[batch, n_nuclei]``.
    eps : float, optional
        Minimum electron-nuclear distance used for numerical safety.

    Returns
    -------
    torch.Tensor
        Electron-nuclear attraction values with shape ``[batch]``.
    """

    if positions.ndim != 3:
        raise ValueError("positions must have shape [batch, n_electrons, spatial_dim]")
    if nuclear_positions.ndim == 2:
        nuclear_positions = nuclear_positions.unsqueeze(0).expand(positions.shape[0], -1, -1)
    if nuclear_charges.ndim == 1:
        nuclear_charges = nuclear_charges.unsqueeze(0).expand(positions.shape[0], -1)
    if nuclear_positions.shape[0] != positions.shape[0] or nuclear_positions.shape[-1] != positions.shape[-1]:
        raise ValueError("nuclear_positions must broadcast to [batch, n_nuclei, spatial_dim]")
    if nuclear_charges.shape != nuclear_positions.shape[:2]:
        raise ValueError("nuclear_charges must broadcast to [batch, n_nuclei]")
    disp = positions.unsqueeze(2) - nuclear_positions.unsqueeze(1)
    dist = torch.linalg.norm(disp, dim=-1).clamp_min(eps)
    output = -(nuclear_charges.unsqueeze(1) / dist).sum(dim=(1, 2))
    assert output.shape == (positions.shape[0],)
    return output


class ElectronicPotential(nn.Module):
    """Evaluate potential energy for an `ElectronicSystem`.

    Parameters
    ----------
    system : ElectronicSystem or None, optional
        Default system metadata used when a batch does not provide a system.
    eps : float, optional
        Distance floor for Coulomb terms.
    """

    def __init__(self, system: ElectronicSystem | None = None, eps: float = 1e-12) -> None:
        super().__init__()
        self.system = system
        self.eps = eps

    def forward(self, batch: ElectronBatch) -> torch.Tensor:
        """Return potential energy for a batch.

        Parameters
        ----------
        batch : ElectronBatch
            Electron batch with positions shaped ``[batch, n_electrons,
            spatial_dim]`` after flattening.

        Returns
        -------
        torch.Tensor
            Potential energy values with shape ``[batch]``.
        """

        batch = batch.flatten_samples()
        system = batch.system or self.system or ElectronicSystem()
        output = torch.zeros(batch.batch_size, device=batch.device, dtype=batch.dtype)
        if system.harmonic_omega is not None:
            output = output + harmonic_trap_potential(batch.positions, omega=float(system.harmonic_omega))
        include_repulsion = bool(system.include_electron_electron) or not system.is_toy_harmonic
        if include_repulsion:
            output = output + electron_electron_repulsion(batch.positions, eps=self.eps)
        if system.nuclear_positions is not None and system.nuclear_charges is not None:
            attraction = electron_nuclear_attraction(
                batch.positions,
                system.nuclear_positions.to(device=batch.device, dtype=batch.dtype),
                system.nuclear_charges.to(device=batch.device, dtype=batch.dtype),
                eps=self.eps,
            )
            output = output + attraction
        assert output.shape == (batch.batch_size,)
        return output
