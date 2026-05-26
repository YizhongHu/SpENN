"""Local-energy evaluation helpers."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data_structures.batch import ElectronBatch
from spenn.physics.kinetic import kinetic_energy_from_logabs
from spenn.physics.potential import ElectronicPotential


class LocalEnergyCalculator(nn.Module):
    """Compute local energies from a black-box model."""

    def __init__(self, potential: ElectronicPotential | None = None) -> None:
        super().__init__()
        self.potential = potential or ElectronicPotential()

    def kinetic(self, model, batch: ElectronBatch) -> torch.Tensor:
        return kinetic_energy_from_logabs(model, batch)

    def potential_energy(self, batch: ElectronBatch) -> torch.Tensor:
        return self.potential(batch)

    def forward(self, model, batch: ElectronBatch) -> torch.Tensor:
        return torch.nan_to_num(self.kinetic(model, batch) + self.potential_energy(batch))
