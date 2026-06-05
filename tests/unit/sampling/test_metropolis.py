"""Unit tests for Metropolis sampling helpers."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data import ElectronBatch, Walkers, WavefunctionOutput
from spenn.physics.systems import ElectronicSystem
from spenn.sampling.metropolis import MetropolisSampler
from spenn.sampling.moves import GaussianMove


class ShiftMove(nn.Module):
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return positions + 1.0


class NoGradLinearModel(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        assert not torch.is_grad_enabled()
        logabs = batch.positions.sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def test_metropolis_sampler_uses_logabs_ratio_caches_values_and_preserves_system() -> None:
    system = ElectronicSystem(n_electrons=2, spatial_dim=1, dtype=torch.float64)
    sampler = MetropolisSampler(move=ShiftMove(), n_walkers=3, dtype=torch.float64)
    walkers = sampler.initialize(system=system, n_walkers=3, device="cpu")
    spins = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    walkers = Walkers(positions=torch.zeros_like(walkers.positions), spins=spins, aux=walkers.aux)

    stepped = sampler.step(NoGradLinearModel(), walkers)

    assert stepped.aux["system"] is system
    assert torch.equal(stepped.positions, torch.ones_like(stepped.positions))
    assert torch.equal(stepped.spins, spins)
    assert torch.equal(stepped.logabs, torch.full((3,), 2.0, dtype=torch.float64))
    assert torch.equal(stepped.sign, torch.ones(3, dtype=torch.float64))
    assert torch.equal(stepped.aux["accepted"], torch.ones(3, dtype=torch.bool))
    assert sampler.acceptance_rate == 1.0


def test_gaussian_single_electron_move_preserves_shape_and_changes_one_electron_per_walker() -> None:
    torch.manual_seed(0)
    positions = torch.zeros(5, 3, 2, dtype=torch.float64)
    walkers = Walkers(positions=positions)
    move = GaussianMove(step_size=0.1, move_all=False)

    proposals, log_q_ratio = move.propose(walkers)
    changed = (proposals != positions).any(dim=-1)

    assert proposals.shape == positions.shape
    assert log_q_ratio.shape == (5,)
    assert torch.allclose(log_q_ratio, torch.zeros(5, dtype=torch.float64))
    assert torch.equal(changed.sum(dim=1), torch.ones(5, dtype=torch.int64))
