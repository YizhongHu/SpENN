"""Physics-side cusp conventions and checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from spenn.data_structures.batch import ElectronBatch
from spenn.utils.tensor_utils import pairwise_distances


@dataclass
class CuspCoefficients:
    same_spin: float = 0.5
    opposite_spin: float = 0.25


class ElectronElectronCusp(nn.Module):
    """Smooth, differentiable electron-electron cusp contribution."""

    def __init__(
        self,
        enabled: bool = True,
        same_spin_coefficient: float | None = None,
        opposite_spin_coefficient: float | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.same_spin_coefficient = CuspCoefficients().same_spin if same_spin_coefficient is None else same_spin_coefficient
        self.opposite_spin_coefficient = (
            CuspCoefficients().opposite_spin if opposite_spin_coefficient is None else opposite_spin_coefficient
        )
        self.eps = eps

    def forward(self, batch: ElectronBatch) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(batch.batch_size, device=batch.device, dtype=batch.dtype)

        distances = pairwise_distances(batch.positions, eps=self.eps).squeeze(-1)
        if batch.spins is None:
            coefficients = torch.full_like(distances, self.opposite_spin_coefficient)
        else:
            spins = batch.spins.unsqueeze(2)
            same_spin = (spins == spins.transpose(1, 2)).to(distances.dtype)
            coefficients = same_spin * self.same_spin_coefficient + (1.0 - same_spin) * self.opposite_spin_coefficient
        tri = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
        contribution = -coefficients * torch.exp(-distances)
        return contribution.masked_fill(~tri, 0.0).sum(dim=(1, 2))
