"""Hooke atom exact-benchmark tests."""

from __future__ import annotations

import torch

from experiments.hooke.analytic import HookeExactWavefunction, hooke_spin_labels
from spenn.data.batch import ElectronBatch
from spenn.physics.hamiltonian import ElectronicHamiltonian
from spenn.physics.potential import ElectronicPotential
from spenn.physics.systems import ElectronicSystem, make_hooke_system


def test_hooke_potential_includes_coulomb_repulsion_only_when_enabled() -> None:
    positions = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=torch.float64)
    hooke = make_hooke_system("singlet", dtype=torch.float64)
    toy = ElectronicSystem(n_electrons=2, spatial_dim=3, harmonic_omega=0.5, dtype=torch.float64)

    hooke_value = ElectronicPotential(system=hooke)(ElectronBatch(positions=positions, system=hooke))
    toy_value = ElectronicPotential(system=toy)(ElectronBatch(positions=positions, system=toy))

    assert torch.allclose(hooke_value, torch.tensor([1.0], dtype=torch.float64))
    assert torch.allclose(toy_value, torch.tensor([0.5], dtype=torch.float64))


def test_exact_hooke_singlet_has_constant_local_energy() -> None:
    system = make_hooke_system("singlet", dtype=torch.float64)
    model = HookeExactWavefunction("singlet")
    hamiltonian = ElectronicHamiltonian(system=system)
    positions = torch.tensor(
        [
            [[0.35, -0.20, 0.10], [-0.45, 0.15, 0.30]],
            [[-0.15, 0.40, -0.35], [0.25, -0.30, 0.50]],
        ],
        dtype=torch.float64,
    )
    batch = ElectronBatch(
        positions=positions,
        system=system,
        spins=hooke_spin_labels("singlet", n_walkers=positions.shape[0], dtype=torch.float64),
    )

    local_energy = hamiltonian.local_energy(model, batch)

    assert local_energy.shape == (2,)
    assert torch.allclose(local_energy, torch.full((2,), 2.0, dtype=torch.float64), atol=1.0e-10)


def test_exact_hooke_triplet_has_constant_local_energy_and_antisymmetry() -> None:
    system = make_hooke_system("triplet", dtype=torch.float64)
    model = HookeExactWavefunction("triplet")
    hamiltonian = ElectronicHamiltonian(system=system)
    positions = torch.tensor(
        [
            [[0.35, -0.20, 0.70], [-0.45, 0.15, -0.30]],
            [[-0.15, 0.40, -0.35], [0.25, -0.30, 0.55]],
        ],
        dtype=torch.float64,
    )
    batch = ElectronBatch(
        positions=positions,
        system=system,
        spins=hooke_spin_labels("triplet", n_walkers=positions.shape[0], dtype=torch.float64),
    )
    swapped = ElectronBatch(positions=positions[:, [1, 0]], system=system, spins=batch.spins)

    local_energy = hamiltonian.local_energy(model, batch)
    output = model(batch)
    swapped_output = model(swapped)

    assert local_energy.shape == (2,)
    assert torch.allclose(local_energy, torch.full((2,), 1.25, dtype=torch.float64), atol=1.0e-10)
    assert torch.allclose(output.logabs, swapped_output.logabs)
    assert torch.equal(output.sign, -swapped_output.sign)
