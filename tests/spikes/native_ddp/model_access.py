"""Raw semantic-module versus DDP-wrapper access split."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


class SemanticWavefunction(nn.Module):
    """Minimal differentiable scalar wavefunction used by the spike."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.25], dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor([-0.10], dtype=torch.float64))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return one log-amplitude per leading sample."""

        if coordinates.ndim != 2:
            raise ValueError("SemanticWavefunction expects [sample, coordinate] input")
        return coordinates.sum(dim=1) * self.weight + self.bias


@dataclass(frozen=True)
class ModelAccess:
    """Typed handles preventing callers from reaching through ``.module``."""

    raw_model: SemanticWavefunction
    ddp_model: DistributedDataParallel

    @classmethod
    def create(cls, raw_model: SemanticWavefunction) -> "ModelAccess":
        """Wrap exactly the raw model once for score-function backward."""

        return cls(
            raw_model=raw_model,
            ddp_model=DistributedDataParallel(raw_model, device_ids=None, broadcast_buffers=False),
        )

    def coordinate_forward(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run coordinate autograd on the raw semantic module only."""

        coordinates = coordinates.detach().clone().requires_grad_(True)
        logabs = self.raw_model(coordinates)
        coordinate_gradient = torch.autograd.grad(
            logabs.sum(), coordinates, create_graph=True, retain_graph=False
        )[0]
        return logabs, coordinate_gradient

    def score_forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Run the sole DDP-wrapped forward used by the parameter update."""

        return self.ddp_model(coordinates)


__all__ = ["ModelAccess", "SemanticWavefunction"]
