"""Unit tests for Metropolis sampling helpers."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, Walkers, WavefunctionOutput
from spenn.sampling import MALASampler
from spenn.sampling.metropolis import MetropolisSampler
from spenn.sampling.moves import GaussianMove


class ShiftMove(nn.Module):
    def propose(self, walkers: Walkers) -> tuple[torch.Tensor, torch.Tensor]:
        positions = walkers.positions + 1.0
        log_q_ratio = torch.zeros(walkers.batch_size, device=walkers.device, dtype=walkers.dtype)
        return positions, log_q_ratio


class NoGradLinearModel(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        assert not torch.is_grad_enabled()
        logabs = batch.positions.sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class QuadraticLogAbsModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.saw_position_grad = False

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        if torch.is_grad_enabled() and batch.positions.requires_grad:
            self.saw_position_grad = True
        logabs = -0.5 * batch.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def test_metropolis_sampler_uses_logabs_ratio_and_caches_values() -> None:
    sampler = MetropolisSampler(move=ShiftMove(), n_walkers=3, n_electrons=2, spatial_dim=1, dtype=torch.float64)
    walkers = sampler.initialize(n_walkers=3, device="cpu")
    spins = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    walkers = Walkers(positions=torch.zeros_like(walkers.positions), spins=spins, aux=walkers.aux)

    stepped = sampler.step(NoGradLinearModel(), walkers)

    assert torch.equal(stepped.positions, torch.ones_like(stepped.positions))
    assert torch.equal(stepped.spins, spins)
    assert torch.equal(stepped.logabs, torch.full((3,), 2.0, dtype=torch.float64))
    assert torch.equal(stepped.sign, torch.ones(3, dtype=torch.float64))
    assert torch.equal(stepped.aux["accepted"], torch.ones(3, dtype=torch.bool))
    assert sampler.acceptance_rate == 1.0


def test_metropolis_initialize_builds_spins_from_partition() -> None:
    sampler = MetropolisSampler(n_walkers=5, n_electrons=2, spatial_dim=3, n_up=1, n_down=1, dtype=torch.float64)

    walkers = sampler.initialize(device="cpu")

    assert walkers.positions.shape == (5, 2, 3)
    assert walkers.spins is not None
    assert torch.equal(walkers.spins, torch.tensor([[1.0, -1.0]], dtype=torch.float64).expand(5, -1))


def test_metropolis_initialize_without_partition_has_no_spins() -> None:
    sampler = MetropolisSampler(n_walkers=4, n_electrons=2, spatial_dim=2, dtype=torch.float64)

    walkers = sampler.initialize(device="cpu")

    assert walkers.positions.shape == (4, 2, 2)
    assert walkers.spins is None


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


def test_mala_sampler_uses_logabs_gradients_and_caches_valid_walkers() -> None:
    torch.manual_seed(0)
    model = QuadraticLogAbsModel()
    sampler = MALASampler(step_size=0.05, n_walkers=4, n_electrons=2, spatial_dim=1, dtype=torch.float64)
    walkers = Walkers(positions=torch.zeros(4, 2, 1, dtype=torch.float64))

    stepped = sampler.step(model, walkers)

    assert model.saw_position_grad
    assert stepped.positions.shape == walkers.positions.shape
    assert stepped.logabs is not None and stepped.logabs.shape == (4,)
    assert stepped.sign is not None and stepped.sign.shape == (4,)
    assert stepped.aux["accepted"].shape == (4,)
    assert stepped.aux["log_accept_ratio"].shape == (4,)
    assert torch.isfinite(stepped.logabs).all()
    assert 0.0 <= sampler.acceptance_rate <= 1.0
