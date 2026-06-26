"""Potential-energy Hamiltonian terms."""

from __future__ import annotations

import torch

from spenn.data.batch import ElectronBatch, pairwise_distances
from spenn.physics.hamiltonian import LocalEnergyResult


class HarmonicTrap:
    """Hamiltonian term for a harmonic confinement potential.

    .. math:: V_\\mathrm{trap} = \\tfrac{1}{2}\\omega^2 \\sum_i r_i^2

    Parameters
    ----------
    omega : float, optional
        Trap frequency.
    """

    name = "harmonic_trap"

    def __init__(self, omega: float = 1.0) -> None:
        self.omega = omega

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        positions = batch.flatten_samples().positions
        if positions.ndim != 3:
            raise ValueError("positions must have shape [batch, n_electrons, spatial_dim]")
        value = 0.5 * (self.omega**2) * positions.square().sum(dim=(1, 2))
        if value.shape != (positions.shape[0],):
            raise ValueError(f"harmonic-trap energy must have shape {(positions.shape[0],)}, got {tuple(value.shape)}")
        return LocalEnergyResult(total=value, terms={self.name: value})


class ElectronElectronInteraction:
    """Hamiltonian term for Coulomb electron-electron repulsion.

    .. math:: V_\\mathrm{ee} = \\sum_{i<j} \\frac{1}{r_{ij}}

    Parameters
    ----------
    eps : float, optional
        Minimum pair distance used for numerical safety.
    """

    name = "electron_electron"

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        positions = batch.flatten_samples().positions
        if positions.ndim != 3:
            raise ValueError("positions must have shape [batch, n_electrons, spatial_dim]")
        distances = pairwise_distances(positions, eps=self.eps).squeeze(-1)
        expected_distances = (positions.shape[0], positions.shape[1], positions.shape[1])
        if distances.shape != expected_distances:
            raise ValueError(f"pairwise distances must have shape {expected_distances}, got {tuple(distances.shape)}")
        tri = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
        value = distances.reciprocal().masked_fill(~tri, 0.0).sum(dim=(1, 2))
        if value.shape != (positions.shape[0],):
            raise ValueError(f"electron-electron energy must have shape {(positions.shape[0],)}, got {tuple(value.shape)}")
        return LocalEnergyResult(total=value, terms={self.name: value})


class ElectronNucleusInteraction:
    """Hamiltonian term for Coulomb electron-nucleus attraction.

    .. math:: V_\\mathrm{en} = -\\sum_{i,A} \\frac{Z_A}{|r_i - R_A|}

    Parameters
    ----------
    nuclear_positions : torch.Tensor
        Nuclear coordinates with shape ``[n_nuclei, spatial_dim]``.
    nuclear_charges : torch.Tensor
        Nuclear charges with shape ``[n_nuclei]``.
    eps : float, optional
        Minimum electron-nucleus distance used for numerical safety.
    """

    name = "electron_nucleus"

    def __init__(
        self,
        nuclear_positions: torch.Tensor,
        nuclear_charges: torch.Tensor,
        eps: float = 1e-12,
    ) -> None:
        self.nuclear_positions = torch.as_tensor(nuclear_positions)
        self.nuclear_charges = torch.as_tensor(nuclear_charges)
        if self.nuclear_positions.ndim != 2:
            raise ValueError("nuclear_positions must have shape [n_nuclei, spatial_dim]")
        if self.nuclear_charges.ndim != 1:
            raise ValueError("nuclear_charges must have shape [n_nuclei]")
        if self.nuclear_positions.shape[0] != self.nuclear_charges.shape[0]:
            raise ValueError("nuclear_positions and nuclear_charges must agree on n_nuclei")
        self.eps = eps

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        positions = batch.flatten_samples().positions
        if positions.ndim != 3:
            raise ValueError("positions must have shape [batch, n_electrons, spatial_dim]")
        nuclear_positions = self.nuclear_positions.to(device=positions.device, dtype=positions.dtype)
        nuclear_charges = self.nuclear_charges.to(device=positions.device, dtype=positions.dtype)
        if nuclear_positions.shape[-1] != positions.shape[-1]:
            raise ValueError("nuclear_positions spatial dimension must match electron positions")
        disp = positions.unsqueeze(2) - nuclear_positions.unsqueeze(0).unsqueeze(0)
        dist = torch.linalg.norm(disp, dim=-1).clamp_min(self.eps)
        value = -(nuclear_charges.view(1, 1, -1) / dist).sum(dim=(1, 2))
        if value.shape != (positions.shape[0],):
            raise ValueError(f"electron-nucleus energy must have shape {(positions.shape[0],)}, got {tuple(value.shape)}")
        return LocalEnergyResult(total=value, terms={self.name: value})


__all__ = [
    "ElectronElectronInteraction",
    "ElectronNucleusInteraction",
    "HarmonicTrap",
]
