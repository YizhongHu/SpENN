"""Physics, local-energy, and VMC-loss sanity tests."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.losses.vmc import VMCLoss
from spenn.nn.cusp import ElectronElectronCusp
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


class CuspGaussianOutputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
        self.cusp = ElectronElectronCusp(
            same_spin_coefficient=0.25,
            opposite_spin_coefficient=0.5,
            range_parameter=0.5,
            eps=1.0e-12,
        )

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = self.cusp(batch) - self.alpha * batch.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class FixedHamiltonian:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values

    def local_energy(self, model, batch: ElectronBatch) -> torch.Tensor:
        return self.values


class FixedLogAbsModel(nn.Module):
    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        self.logabs = nn.Parameter(values.clone())

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = self.logabs[: batch.batch_size]
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


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


def test_three_electron_hooke_local_energy_matches_gaussian_formula() -> None:
    omega = 0.5
    alpha = 0.2
    system = ElectronicSystem(
        n_electrons=3,
        spatial_dim=3,
        harmonic_omega=omega,
        include_electron_electron=True,
        n_up=2,
        n_down=1,
    )
    positions = torch.tensor(
        [
            [[0.25, -0.50, 0.75], [1.25, 0.50, -0.25], [-0.75, 0.10, 0.40]],
            [[-1.00, 0.25, 0.50], [0.30, -0.80, 1.10], [0.90, 0.70, -0.60]],
        ],
        dtype=torch.float64,
    )
    batch = ElectronBatch(
        positions=positions,
        system=system,
        spins=torch.tensor([[1.0, 1.0, -1.0], [1.0, 1.0, -1.0]], dtype=torch.float64),
    )
    hamiltonian = ElectronicHamiltonian(system=system)

    local_energy = hamiltonian.local_energy(GaussianTensorModel(alpha=alpha), batch)

    squared_radius = positions.square().sum(dim=(1, 2))
    kinetic = alpha * system.n_electrons * system.spatial_dim - 2.0 * alpha**2 * squared_radius
    harmonic = 0.5 * omega**2 * squared_radius
    repulsion = sum(
        torch.linalg.norm(positions[:, i] - positions[:, j], dim=-1).reciprocal()
        for i, j in ((0, 1), (0, 2), (1, 2))
    )
    expected = kinetic + harmonic + repulsion
    assert local_energy.shape == (2,)
    assert torch.allclose(local_energy, expected, atol=1.0e-12)


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


def test_cusp_local_energy_has_finite_second_derivatives_with_pair_diagonal() -> None:
    system = ElectronicSystem(n_electrons=2, spatial_dim=3, harmonic_omega=0.5, include_electron_electron=True)
    batch = ElectronBatch(
        positions=torch.tensor([[[0.25, -0.1, 0.3], [-0.35, 0.4, -0.2]]], dtype=torch.float64),
        system=system,
        spins=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
    )
    model = CuspGaussianOutputModel()
    hamiltonian = ElectronicHamiltonian(system=system)

    local_energy = hamiltonian.local_energy(model, batch)
    local_energy.mean().backward()

    assert local_energy.shape == (1,)
    assert torch.all(torch.isfinite(local_energy))
    assert model.alpha.grad is not None
    assert torch.isfinite(model.alpha.grad)


def test_vmc_loss_returns_score_function_objective_and_detached_metrics() -> None:
    local_energy = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
    logabs = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    batch = ElectronBatch(positions=torch.zeros(3, 2, 1, dtype=torch.float64))
    model = FixedLogAbsModel(logabs)

    loss, metrics = VMCLoss()(model=model, hamiltonian=FixedHamiltonian(local_energy), batch=batch)
    expected_loss = 2.0 * ((local_energy.detach() - local_energy.detach().mean()) * logabs).mean()

    assert torch.equal(loss, expected_loss)
    assert torch.equal(metrics["energy"], torch.tensor(3.0, dtype=torch.float64))
    assert torch.equal(metrics["variance"], torch.tensor(8.0 / 3.0, dtype=torch.float64))
    assert not metrics["energy"].requires_grad
    assert not metrics["variance"].requires_grad
