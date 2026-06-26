"""Tiny trainable wavefunctions for training smoke tests."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from spenn.data.batch import ElectronBatch, WavefunctionOutput


class TrainableHookeSingletAnsatz(nn.Module):
    """Two-electron Hooke singlet ansatz with learnable width and Jastrow.

    Mirrors `spenn.physics.hooke.HookeSingletExact` with trainable parameters::

        psi(r1, r2) = (1 + beta * r12) * exp(-alpha * (r1^2 + r2^2))

    ``alpha`` and ``beta`` are softplus transforms of unconstrained float64
    parameters, keeping them positive so ``logabs`` stays finite.
    """

    def __init__(self, raw_alpha: float = 0.0, raw_beta: float = 0.0) -> None:
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.tensor(float(raw_alpha), dtype=torch.float64))
        self.raw_beta = nn.Parameter(torch.tensor(float(raw_beta), dtype=torch.float64))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Return the signed log-amplitude for a batch of configurations."""

        pos = batch.flatten_samples().positions  # [B, 2, 3]
        alpha = F.softplus(self.raw_alpha)
        beta = F.softplus(self.raw_beta)
        r1_sq = pos[:, 0, :].square().sum(-1)
        r2_sq = pos[:, 1, :].square().sum(-1)
        r12 = (pos[:, 0, :] - pos[:, 1, :]).norm(dim=-1)
        logabs = torch.log1p(beta * r12) - alpha * (r1_sq + r2_sq)
        sign = torch.ones_like(logabs)
        return WavefunctionOutput(logabs=logabs, sign=sign)


__all__ = ["TrainableHookeSingletAnsatz"]
