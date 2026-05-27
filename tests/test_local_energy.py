"""Physics, local-energy, and VMC-loss sanity tests."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.losses.vmc import VMCLoss
from spenn.physics.hamiltonian import ElectronicHamiltonian
from spenn.physics.kinetic import kinetic_energy_from_logabs
from spenn.physics.potential import (
    electron_electron_repulsion,
    electron_nuclear_attraction,
    harmonic_trap_potential,
)
from spenn.physics.systems import ElectronicSystem


class GaussianTensorModel(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float64)

    def forward(self, batch: ElectronBatch) -> torch.Tensor:
        return -self.alpha * batch.positions.square().sum(dim=(1, 2))


class TrainableGaussianOutputModel(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float64))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = -self.alpha * batch.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class FixedHamiltonian:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values

    def local_energy(self, model, batch: ElectronBatch) -> torch.Tensor:
        return self.values


def test_potential_terms_match_direct_hand_calculations() -> None:
    positions = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.5, -1.0], [2.5, 1.0]],
        ],
        dtype=torch.float64,
    )
    nuclei = torch.tensor([[0.0, -1.0], [2.0, 0.0]], dtype=torch.float64)
    charges = torch.tensor([2.0, 0.5], dtype=torch.float64)

    harmonic = harmonic_trap_potential(positions, omega=1.5)
    repulsion = electron_electron_repulsion(positions)
    attraction = electron_nuclear_attraction(positions, nuclei, charges)
    batched_attraction = electron_nuclear_attraction(
        positions,
        nuclei.unsqueeze(0).expand(positions.shape[0], -1, -1),
        charges.unsqueeze(0).expand(positions.shape[0], -1),
    )

    expected_harmonic = 0.5 * (1.5**2) * positions.square().sum(dim=(1, 2))
    expected_repulsion = torch.linalg.norm(positions[:, 0] - positions[:, 1], dim=-1).reciprocal()
    expected_attraction = -(
        charges.view(1, 1, -1)
        / torch.linalg.norm(positions.unsqueeze(2) - nuclei.view(1, 1, 2, 2), dim=-1)
    ).sum(dim=(1, 2))

    assert torch.allclose(harmonic, expected_harmonic)
    assert torch.allclose(repulsion, expected_repulsion)
    assert torch.allclose(attraction, expected_attraction)
    assert torch.allclose(batched_attraction, expected_attraction)


def test_autograd_kinetic_matches_gaussian_logabs_formula() -> None:
    positions = torch.tensor(
        [
            [[1.0, 2.0], [0.5, -1.0]],
            [[-1.5, 0.25], [2.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    alpha = 0.3
    batch = ElectronBatch(positions=positions)

    kinetic = kinetic_energy_from_logabs(GaussianTensorModel(alpha), batch)

    n_electrons = positions.shape[1]
    spatial_dim = positions.shape[2]
    expected = alpha * n_electrons * spatial_dim - 2.0 * alpha**2 * positions.square().sum(dim=(1, 2))
    assert torch.allclose(kinetic, expected)


def test_harmonic_oscillator_ground_state_has_constant_local_energy() -> None:
    omega = 1.7
    system = ElectronicSystem(n_electrons=2, spatial_dim=3, harmonic_omega=omega)
    positions = torch.tensor(
        [
            [[1.0, 0.0, -1.0], [0.5, 2.0, 0.25]],
            [[-0.5, 1.5, 2.5], [1.0, -2.0, 0.5]],
        ],
        dtype=torch.float64,
    )
    batch = ElectronBatch(positions=positions, system=system)
    hamiltonian = ElectronicHamiltonian(system=system)

    local_energy = hamiltonian.local_energy(GaussianTensorModel(alpha=omega / 2.0), batch)

    expected = torch.full((2,), system.n_electrons * system.spatial_dim * omega / 2.0, dtype=torch.float64)
    assert torch.allclose(local_energy, expected)


def test_local_energy_accepts_wavefunction_output_and_preserves_parameter_gradients() -> None:
    system = ElectronicSystem(n_electrons=2, spatial_dim=2, harmonic_omega=1.0)
    batch = ElectronBatch(
        positions=torch.tensor([[[1.0, 0.0], [0.0, 2.0]], [[-1.0, 1.0], [2.0, -0.5]]], dtype=torch.float64),
        system=system,
    )
    model = TrainableGaussianOutputModel(alpha=0.25)
    hamiltonian = ElectronicHamiltonian(system=system)

    local_energy = hamiltonian.local_energy(model, batch)
    local_energy.mean().backward()

    assert local_energy.shape == (2,)
    assert torch.all(torch.isfinite(local_energy))
    assert model.alpha.grad is not None
    assert torch.isfinite(model.alpha.grad)


def test_vmc_loss_returns_mean_energy_and_detached_metrics() -> None:
    local_energy = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
    batch = ElectronBatch(positions=torch.zeros(3, 2, 1, dtype=torch.float64))

    loss, metrics = VMCLoss()(model=nn.Identity(), hamiltonian=FixedHamiltonian(local_energy), batch=batch)

    assert torch.equal(loss, local_energy.mean())
    assert torch.equal(metrics["energy"], torch.tensor(3.0, dtype=torch.float64))
    assert torch.equal(metrics["variance"], torch.tensor(8.0 / 3.0, dtype=torch.float64))
    assert not metrics["energy"].requires_grad
    assert not metrics["variance"].requires_grad
