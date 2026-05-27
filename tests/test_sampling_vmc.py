"""Sampler and one-step VMC trainer integration tests."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, Walkers, WavefunctionOutput
from spenn.losses.vmc import VMCLoss
from spenn.physics.hamiltonian import ElectronicHamiltonian
from spenn.physics.systems import ElectronicSystem
from spenn.sampling.metropolis import MetropolisSampler
from spenn.training.trainer import VMCTrainer


class ShiftMove(nn.Module):
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return positions + 1.0


class NoGradLinearModel(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        assert not torch.is_grad_enabled()
        logabs = batch.positions.sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class TrainableGaussianModel(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float64))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = -self.alpha * batch.positions.square().sum(dim=(1, 2))
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


def test_vmc_trainer_one_step_updates_parameter_and_reports_finite_metrics() -> None:
    system = ElectronicSystem(n_electrons=2, spatial_dim=2, harmonic_omega=1.0, dtype=torch.float64)
    walkers = Walkers(
        positions=torch.ones(4, 2, 2, dtype=torch.float64),
        aux={"system": system},
    )
    model = TrainableGaussianModel(alpha=0.1)
    sampler = MetropolisSampler(n_walkers=4, steps_per_iter=1, step_size=0.0, dtype=torch.float64)
    hamiltonian = ElectronicHamiltonian(system=system)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = VMCTrainer(
        model=model,
        sampler=sampler,
        hamiltonian=hamiltonian,
        loss=VMCLoss(),
        optimizer=optimizer,
        system=system,
        walkers=walkers,
        max_steps=1,
    )
    before = model.alpha.detach().clone()

    metrics = trainer.train_step()

    assert trainer.global_step == 1
    assert not torch.equal(model.alpha.detach(), before)
    assert trainer.walkers.positions.shape == (4, 2, 2)
    for key in ("loss", "energy", "variance", "acceptance_rate", "grad_norm", "param_norm"):
        assert key in metrics
        assert torch.isfinite(metrics[key])
