"""Potential-energy terms."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch
from spenn.physics.systems import ElectronicSystem
from spenn.utils.tensor_utils import pairwise_distances


def harmonic_trap_potential(positions: torch.Tensor, omega: float = 1.0) -> torch.Tensor:
    """Harmonic confinement potential for each walker."""

    assert positions.ndim == 3
    output = 0.5 * (omega**2) * positions.square().sum(dim=(1, 2))
    assert output.shape == (positions.shape[0],)
    return output


def electron_electron_repulsion(positions: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Coulomb repulsion summed over unique electron pairs."""

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
    """Electron-nuclear attraction for batched positions."""

    if nuclear_positions.ndim == 2:
        nuclear_positions = nuclear_positions.unsqueeze(0).expand(positions.shape[0], -1, -1)
    if nuclear_charges.ndim == 1:
        nuclear_charges = nuclear_charges.unsqueeze(0).expand(positions.shape[0], -1)
    assert positions.ndim == 3
    assert nuclear_positions.shape[0] == positions.shape[0]
    assert nuclear_positions.shape[-1] == positions.shape[-1]
    assert nuclear_charges.shape == nuclear_positions.shape[:2]
    disp = positions.unsqueeze(2) - nuclear_positions.unsqueeze(1)
    dist = torch.linalg.norm(disp, dim=-1).clamp_min(eps)
    output = -(nuclear_charges.unsqueeze(1) / dist).sum(dim=(1, 2))
    assert output.shape == (positions.shape[0],)
    return output


class ElectronicPotential(nn.Module):
    """Potential energy module that treats the model as a black box."""

    def __init__(self, system: ElectronicSystem | None = None, eps: float = 1e-12, **_: object) -> None:
        super().__init__()
        self.system = system
        self.eps = eps

    def forward(self, batch: ElectronBatch) -> torch.Tensor:
        system = batch.system or self.system or ElectronicSystem()
        if system.is_toy_harmonic:
            return harmonic_trap_potential(batch.positions, omega=system.harmonic_omega)
        repulsion = electron_electron_repulsion(batch.positions, eps=self.eps)
        attraction = torch.zeros_like(repulsion)
        if system.nuclear_positions is not None and system.nuclear_charges is not None:
            attraction = electron_nuclear_attraction(
                batch.positions,
                system.nuclear_positions.to(device=batch.device, dtype=batch.dtype),
                system.nuclear_charges.to(device=batch.device, dtype=batch.dtype),
                eps=self.eps,
            )
        harmonic = harmonic_trap_potential(batch.positions, omega=system.harmonic_omega)
        output = harmonic + repulsion + attraction
        assert output.shape == (batch.batch_size,)
        return output
