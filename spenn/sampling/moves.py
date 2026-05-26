"""Proposal move helpers."""

from __future__ import annotations

import torch
from torch import nn


def gaussian_proposal(positions: torch.Tensor, step_size: float) -> torch.Tensor:
    """Return a Gaussian random-walk proposal."""

    return positions + step_size * torch.randn_like(positions)


class GaussianMove(nn.Module):
    """Gaussian random-walk proposal module."""

    def __init__(self, step_size: float = 0.05, move_all: bool = True) -> None:
        super().__init__()
        self.step_size = step_size
        self.move_all = move_all

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if self.move_all:
            return gaussian_proposal(positions, self.step_size)
        proposals = positions.clone()
        electron_idx = torch.randint(positions.shape[1], (positions.shape[0],), device=positions.device)
        walker_idx = torch.arange(positions.shape[0], device=positions.device)
        proposals[walker_idx, electron_idx] = gaussian_proposal(proposals[walker_idx, electron_idx], self.step_size)
        return proposals
