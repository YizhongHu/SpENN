"""Potential-energy Hamiltonian terms."""

from __future__ import annotations

import torch

from tpen.data.batch import ElectronBatch, nuclear_potential, pairwise_distances
from tpen.physics.hamiltonian import LocalEnergyResult


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
    nuclear_positions : torch.Tensor or None, optional
        Deprecated compatibility metadata. When supplied, it must agree
        exactly with the nuclear positions carried by every evaluated batch.
    nuclear_charges : torch.Tensor or None, optional
        Deprecated compatibility metadata paired with ``nuclear_positions``.
    eps : float, optional
        Minimum electron-nucleus distance used for numerical safety.
    """

    name = "electron_nucleus"

    def __init__(
        self,
        nuclear_positions: torch.Tensor | None = None,
        nuclear_charges: torch.Tensor | None = None,
        eps: float = 1e-12,
    ) -> None:
        if (nuclear_positions is None) != (nuclear_charges is None):
            raise ValueError("nuclear_positions and nuclear_charges must be provided together")
        self.nuclear_positions = None if nuclear_positions is None else torch.as_tensor(nuclear_positions).detach().clone()
        self.nuclear_charges = None if nuclear_charges is None else torch.as_tensor(nuclear_charges).detach().clone()
        if self.nuclear_positions is not None and self.nuclear_positions.ndim != 2:
            raise ValueError("nuclear_positions must have shape [n_nuclei, spatial_dim]")
        if self.nuclear_charges is not None and self.nuclear_charges.ndim != 1:
            raise ValueError("nuclear_charges must have shape [n_nuclei]")
        if self.nuclear_positions is not None and self.nuclear_positions.shape[0] != self.nuclear_charges.shape[0]:
            raise ValueError("nuclear_positions and nuclear_charges must agree on n_nuclei")
        self.eps = eps

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        flat = batch.flatten_samples()
        positions = flat.positions
        if positions.ndim != 3:
            raise ValueError("positions must have shape [batch, n_electrons, spatial_dim]")
        _require_nuclear_context(flat)
        self._validate_legacy_context(flat)
        potential = nuclear_potential(flat, eps=self.eps)
        expected_potential = (positions.shape[0], positions.shape[1])
        if potential.shape != expected_potential:
            raise ValueError(f"electron-nucleus potential must have shape {expected_potential}, got {tuple(potential.shape)}")
        value = -potential.sum(dim=1)
        if value.shape != (positions.shape[0],):
            raise ValueError(f"electron-nucleus energy must have shape {(positions.shape[0],)}, got {tuple(value.shape)}")
        return LocalEnergyResult(total=value, terms={self.name: value})

    def _validate_legacy_context(self, batch: ElectronBatch) -> None:
        """Reject legacy constructor metadata that disagrees with a batch."""

        if self.nuclear_positions is None:
            return
        assert self.nuclear_charges is not None
        positions = self.nuclear_positions.to(device=batch.device, dtype=batch.dtype)
        charges = self.nuclear_charges.to(device=batch.device, dtype=batch.dtype)
        if not torch.equal(positions, batch.nuclear_positions) or not torch.equal(charges, batch.nuclear_charges):
            raise ValueError("legacy ElectronNucleusInteraction nuclear metadata must agree exactly with batch context")


class NucleusNucleusInteraction:
    r"""Hamiltonian term for Born--Oppenheimer nuclear repulsion.

    .. math:: V_\mathrm{nn} = \sum_{A<B} \frac{Z_A Z_B}{|R_A - R_B|}

    Nuclear coordinates and charges are read from the explicit metadata on
    ``ElectronBatch``.  Shared ``[n_nuclei, spatial_dim]`` metadata and
    per-configuration ``[batch, n_nuclei, spatial_dim]`` metadata are both
    supported.
    """

    name = "nucleus_nucleus"

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        """Return the pairwise nuclear repulsion for each batch sample."""

        flat = batch.flatten_samples()
        positions = flat.nuclear_positions
        charges = flat.nuclear_charges
        if positions is None:
            raise ValueError("NucleusNucleusInteraction requires batch.nuclear_positions")
        if charges is None:
            raise ValueError("NucleusNucleusInteraction requires batch.nuclear_charges")

        positions = positions.to(device=flat.device, dtype=flat.dtype)
        charges = charges.to(device=flat.device, dtype=flat.dtype)
        if positions.ndim == 2:
            if positions.shape[-1] != flat.spatial_dim:
                raise ValueError("nuclear positions must match batch spatial dimension")
            positions = positions.unsqueeze(0).expand(flat.batch_size, -1, -1)
        elif positions.ndim == 3:
            if positions.shape[0] != flat.batch_size or positions.shape[-1] != flat.spatial_dim:
                raise ValueError("batched nuclear positions must match batch size and spatial dimension")
        else:
            raise ValueError("nuclear positions must have shape [n_nuclei, dim] or [batch, n_nuclei, dim]")

        if charges.ndim == 1:
            charges = charges.unsqueeze(0).expand(flat.batch_size, -1)
        elif charges.ndim != 2 or charges.shape[0] != flat.batch_size:
            raise ValueError("nuclear charges must have shape [n_nuclei] or [batch, n_nuclei]")
        if positions.shape[1] != charges.shape[1]:
            raise ValueError("nuclear positions and charges must agree on n_nuclei")
        if not torch.isfinite(positions).all() or not torch.isfinite(charges).all():
            raise ValueError("nuclear positions and charges must be finite")

        n_nuclei = positions.shape[1]
        if n_nuclei < 2:
            value = torch.zeros(flat.batch_size, device=flat.device, dtype=flat.dtype)
        else:
            distances = torch.linalg.norm(positions.unsqueeze(2) - positions.unsqueeze(1), dim=-1)
            pair_mask = torch.triu(torch.ones((n_nuclei, n_nuclei), device=flat.device, dtype=torch.bool), diagonal=1)
            if (distances[:, pair_mask] <= 0).any():
                raise ValueError("nuclear positions contain colliding nuclei")
            pair_values = charges.unsqueeze(2) * charges.unsqueeze(1) / distances
            value = pair_values[:, pair_mask].sum(dim=1)

        return LocalEnergyResult(total=value, terms={self.name: value})


def _require_nuclear_context(batch: ElectronBatch) -> None:
    """Require the typed nuclear context owned by an electron batch."""

    if batch.nuclear_positions is None:
        raise ValueError("ElectronNucleusInteraction requires batch.nuclear_positions")
    if batch.nuclear_charges is None:
        raise ValueError("ElectronNucleusInteraction requires batch.nuclear_charges")


__all__ = [
    "ElectronElectronInteraction",
    "ElectronNucleusInteraction",
    "HarmonicTrap",
    "NucleusNucleusInteraction",
]
