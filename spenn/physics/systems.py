"""System definitions and molecule/toy-problem metadata."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch


@dataclass
class ElectronicSystem:
    """Minimal electronic or toy system description."""

    n_electrons: int = 2
    spatial_dim: int = 3
    nuclear_positions: torch.Tensor | None = None
    nuclear_charges: torch.Tensor | None = None
    harmonic_omega: float = 1.0
    device: torch.device | str | None = None
    dtype: torch.dtype | str | None = torch.float64
    name: str = "toy"
    aux: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.nuclear_positions is not None and not isinstance(self.nuclear_positions, torch.Tensor):
            self.nuclear_positions = torch.as_tensor(self.nuclear_positions, dtype=torch.float64)
        if self.nuclear_charges is not None and not isinstance(self.nuclear_charges, torch.Tensor):
            self.nuclear_charges = torch.as_tensor(self.nuclear_charges, dtype=torch.float64)
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "ElectronicSystem":
        return replace(
            self,
            nuclear_positions=None if self.nuclear_positions is None else self.nuclear_positions.to(device=device, dtype=dtype),
            nuclear_charges=None if self.nuclear_charges is None else self.nuclear_charges.to(device=device, dtype=dtype),
            device=device or self.device,
            dtype=dtype or self.dtype,
        )

    @property
    def is_toy_harmonic(self) -> bool:
        return self.nuclear_positions is None or self.nuclear_charges is None
