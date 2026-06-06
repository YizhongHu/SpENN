"""Integration tests for the VMC trainer loop."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, Walkers, WavefunctionOutput
from spenn.losses import VMCLoss
from spenn.physics.hamiltonian import ElectronicHamiltonian
from spenn.physics.systems import ElectronicSystem
from spenn.sampling.metropolis import MetropolisSampler
from spenn.training.trainer import VMCTrainer


class TrainableGaussianModel(nn.Module):
    """Tiny trainable wavefunction used by trainer smoke tests."""

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float64))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Return a signed Gaussian log-amplitude."""

        logabs = -self.alpha * batch.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def test_vmc_trainer_one_step_updates_parameter_and_reports_finite_metrics() -> None:
    system = ElectronicSystem(n_electrons=2, spatial_dim=2, harmonic_omega=1.0, dtype=torch.float64)
    walkers = Walkers(
        positions=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 2.0]],
                [[0.5, 1.5], [1.0, -0.5]],
                [[-1.0, 0.25], [0.75, 0.5]],
                [[1.5, -0.25], [-0.5, 1.0]],
            ],
            dtype=torch.float64,
        ),
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
