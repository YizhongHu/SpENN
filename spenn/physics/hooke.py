"""Exact two-electron Hooke atom wavefunction references."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, WavefunctionOutput


class HookeSingletExact(nn.Module):
    """Exact spatial wavefunction for the two-electron Hooke singlet (omega=1/2).

    psi(r1, r2) = (1 + r12/2) * exp(-r1^2/4 - r2^2/4)
    Exact energy: E = 2.0
    """

    omega: float = 0.5
    exact_energy: float = 2.0

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        pos = batch.flatten_samples().positions  # [B, 2, 3]
        r1_sq = pos[:, 0, :].square().sum(-1)
        r2_sq = pos[:, 1, :].square().sum(-1)
        r12 = (pos[:, 0, :] - pos[:, 1, :]).norm(dim=-1)
        logabs = torch.log1p(0.5 * r12) - 0.25 * (r1_sq + r2_sq)
        sign = torch.ones_like(logabs)
        return WavefunctionOutput(logabs=logabs, sign=sign)


class HookeTripletExact(nn.Module):
    """Exact spatial wavefunction for the two-electron Hooke triplet (omega=1/4).

    psi(r1, r2) = (z1 - z2) * (1 + r12/4) * exp(-r1^2/8 - r2^2/8)
    Exact energy: E = 1.25

    Tests must sample away from the node z1 = z2.
    """

    omega: float = 0.25
    exact_energy: float = 1.25

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        pos = batch.flatten_samples().positions  # [B, 2, 3]
        r1_sq = pos[:, 0, :].square().sum(-1)
        r2_sq = pos[:, 1, :].square().sum(-1)
        r12 = (pos[:, 0, :] - pos[:, 1, :]).norm(dim=-1)
        z_diff = pos[:, 0, 2] - pos[:, 1, 2]
        sign = torch.sign(z_diff)
        logabs = torch.log(torch.abs(z_diff)) + torch.log1p(0.25 * r12) - 0.125 * (r1_sq + r2_sq)
        return WavefunctionOutput(logabs=logabs, sign=sign)


__all__ = ["HookeSingletExact", "HookeTripletExact"]
