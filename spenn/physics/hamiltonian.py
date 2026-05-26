"""Hamiltonian interfaces."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data_structures.batch import ElectronBatch
from spenn.physics.kinetic import LogAbsKineticEnergy
from spenn.physics.potential import ElectronicPotential
from spenn.physics.systems import ElectronicSystem


class ElectronicHamiltonian(nn.Module):
    """Minimal black-box Hamiltonian used for phase 1."""

    def __init__(
        self,
        system: ElectronicSystem | None = None,
        kinetic: nn.Module | None = None,
        potential: ElectronicPotential | None = None,
        name: str = "electronic",
        **_: object,
    ) -> None:
        super().__init__()
        self.name = name
        self.system = system or ElectronicSystem()
        self.kinetic_module = kinetic or LogAbsKineticEnergy()
        self.potential = potential or ElectronicPotential(system=self.system)
        if getattr(self.potential, "system", None) is None:
            self.potential.system = self.system

    def kinetic(self, model, batch: ElectronBatch) -> torch.Tensor:
        return self.kinetic_module(model, batch)

    def potential_energy(self, batch: ElectronBatch) -> torch.Tensor:
        return self.potential(batch)

    def local_energy(self, model, batch: ElectronBatch) -> torch.Tensor:
        return torch.nan_to_num(self.kinetic(model, batch) + self.potential_energy(batch))

    def forward(self, model, batch: ElectronBatch) -> torch.Tensor:
        return self.local_energy(model, batch)
