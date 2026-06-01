"""System definitions and molecule/toy-problem metadata."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch


@dataclass
class ElectronicSystem:
    """Describe a fixed electronic or toy system.

    Parameters
    ----------
    n_electrons : int, optional
        Number of electrons in each configuration.
    spatial_dim : int, optional
        Spatial dimension of each electron coordinate.
    nuclear_positions : torch.Tensor or None, optional
        Nuclear coordinates with shape ``[n_nuclei, spatial_dim]``.
    nuclear_charges : torch.Tensor or None, optional
        Nuclear charges with shape ``[n_nuclei]``.
    harmonic_omega : float or None, optional
        Harmonic trap frequency. If ``None``, no trap contribution is added.
    include_electron_electron : bool, optional
        Whether to include Coulomb electron-electron repulsion for systems
        without nuclei. Molecular systems with nuclei keep electron-electron
        repulsion regardless of this flag for backward compatibility.
    n_up, n_down : int or None, optional
        Spin partition metadata.
    device : torch.device, str, or None, optional
        Preferred device for initialized tensors.
    dtype : torch.dtype, str, or None, optional
        Preferred floating-point dtype.
    name : str, optional
        Human-readable system name.
    exact_energy : float or None, optional
        Known benchmark energy, when available.
    aux : dict or None, optional
        Additional metadata.
    """

    n_electrons: int = 2
    spatial_dim: int = 3
    nuclear_positions: torch.Tensor | None = None
    nuclear_charges: torch.Tensor | None = None
    harmonic_omega: float | None = 1.0
    include_electron_electron: bool = False
    n_up: int | None = None
    n_down: int | None = None
    device: torch.device | str | None = None
    dtype: torch.dtype | str | None = torch.float64
    name: str = "toy"
    exact_energy: float | None = None
    aux: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.n_electrons <= 0:
            raise ValueError("ElectronicSystem.n_electrons must be positive")
        if self.spatial_dim <= 0:
            raise ValueError("ElectronicSystem.spatial_dim must be positive")
        if self.n_up is not None and self.n_down is not None and self.n_up + self.n_down != self.n_electrons:
            raise ValueError("ElectronicSystem.n_up + n_down must equal n_electrons")
        if self.nuclear_positions is not None and not isinstance(self.nuclear_positions, torch.Tensor):
            self.nuclear_positions = torch.as_tensor(self.nuclear_positions, dtype=torch.float64)
        if self.nuclear_charges is not None and not isinstance(self.nuclear_charges, torch.Tensor):
            self.nuclear_charges = torch.as_tensor(self.nuclear_charges, dtype=torch.float64)
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)
        if self.nuclear_positions is not None:
            if self.nuclear_positions.ndim != 2 or self.nuclear_positions.shape[-1] != self.spatial_dim:
                raise ValueError("ElectronicSystem.nuclear_positions must have shape [n_nuclei, spatial_dim]")
        if self.nuclear_charges is not None:
            if self.nuclear_charges.ndim != 1:
                raise ValueError("ElectronicSystem.nuclear_charges must have shape [n_nuclei]")
        if self.nuclear_positions is not None and self.nuclear_charges is not None:
            if self.nuclear_positions.shape[0] != self.nuclear_charges.shape[0]:
                raise ValueError("ElectronicSystem nuclear positions and charges must agree on n_nuclei")

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "ElectronicSystem":
        """Move tensor metadata to a new device or dtype.

        Parameters
        ----------
        device : torch.device, str, or None, optional
            Target device. If ``None``, the existing device metadata is kept.
        dtype : torch.dtype or None, optional
            Target dtype for tensor metadata.

        Returns
        -------
        ElectronicSystem
            System with tensor fields moved to the requested target.
        """

        return replace(
            self,
            nuclear_positions=None if self.nuclear_positions is None else self.nuclear_positions.to(device=device, dtype=dtype),
            nuclear_charges=None if self.nuclear_charges is None else self.nuclear_charges.to(device=device, dtype=dtype),
            device=device or self.device,
            dtype=dtype or self.dtype,
        )

    @property
    def is_toy_harmonic(self) -> bool:
        """Return whether the system has no nuclear metadata.

        Returns
        -------
        bool
            ``True`` when nuclear positions or charges are absent.
        """

        return self.nuclear_positions is None or self.nuclear_charges is None
