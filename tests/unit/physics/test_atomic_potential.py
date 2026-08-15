"""Tests for constructor-owned AtomicConfiguration Coulomb potentials."""

from __future__ import annotations

import pytest
import torch

from tpen.data import AtomicConfiguration
from tpen.data.batch import ElectronBatch
from tpen.data.permutation import Permutation
from tpen.physics.hamiltonian import LocalEnergyResult, local_energy
from tpen.physics.potential import ElectronNucleusPotential, NucleusNucleusPotential


def _atoms(nuclei, charges, dtype=torch.float64) -> AtomicConfiguration:
    return AtomicConfiguration(
        positions=torch.tensor(nuclei, dtype=dtype),
        charges=torch.tensor(charges, dtype=dtype),
    )


def test_electron_nucleus_potential_rejects_non_atomic_configuration() -> None:
    with pytest.raises(TypeError, match="AtomicConfiguration"):
        ElectronNucleusPotential(atoms=object())


def test_nucleus_nucleus_potential_rejects_non_atomic_configuration() -> None:
    with pytest.raises(TypeError, match="AtomicConfiguration"):
        NucleusNucleusPotential(atoms=object())


def test_electron_nucleus_potential_matches_hand_calculation() -> None:
    positions = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[-1.0, 0.5, 0.0], [0.0, 0.0, 2.0]]],
        dtype=torch.float64,
    )
    atoms = _atoms([[0.0, 0.0, 0.0]], [2.0])
    batch = ElectronBatch(positions=positions)

    result = ElectronNucleusPotential(atoms).local_energy(None, batch)

    expected = -(
        atoms.charges.view(1, 1, -1)
        / torch.linalg.norm(positions.unsqueeze(2) - atoms.positions.view(1, 1, 1, 3), dim=-1)
    ).sum(dim=(1, 2))
    assert isinstance(result, LocalEnergyResult)
    torch.testing.assert_close(result.total, expected)
    torch.testing.assert_close(result.terms["electron_nucleus"], expected)


def test_electron_nucleus_potential_permutation_invariant() -> None:
    positions = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]], dtype=torch.float64)
    atoms = _atoms([[0.0, 0.0], [3.0, 0.0]], [2.0, 1.0])
    batch = ElectronBatch(positions=positions)

    result = ElectronNucleusPotential(atoms).local_energy(None, batch)
    permuted = ElectronNucleusPotential(atoms).local_energy(None, batch.permute(Permutation((1, 0))))

    torch.testing.assert_close(result.total, permuted.total)


def test_electron_nucleus_potential_raw_eps_zero_matches_reciprocal() -> None:
    positions = torch.tensor([[[3.0, 0.0]]], dtype=torch.float64)
    atoms = _atoms([[0.0, 0.0]], [2.0])

    result = ElectronNucleusPotential(atoms, eps=0.0).local_energy(None, ElectronBatch(positions=positions))

    assert torch.allclose(result.total, torch.tensor([-2.0 / 3.0], dtype=torch.float64))


def test_electron_nucleus_potential_requires_agreement_with_batch_nuclear_positions() -> None:
    positions = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]], dtype=torch.float64)
    atoms = _atoms([[0.0, 0.0], [3.0, 0.0]], [2.0, 1.0])
    batch = ElectronBatch(
        positions=positions,
        nuclear_positions=atoms.positions,
        nuclear_charges=atoms.charges,
    )

    ElectronNucleusPotential(atoms).local_energy(None, batch)  # agreeing context is fine

    mismatched = ElectronBatch(
        positions=positions,
        nuclear_positions=atoms.positions + 1.0,
        nuclear_charges=atoms.charges,
    )
    with pytest.raises(ValueError, match="agree exactly with batch.nuclear_positions"):
        ElectronNucleusPotential(atoms).local_energy(None, mismatched)


def test_electron_nucleus_potential_requires_agreement_with_batch_nuclear_charges() -> None:
    positions = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    atoms = _atoms([[0.0, 0.0]], [2.0])
    mismatched = ElectronBatch(
        positions=positions,
        nuclear_positions=atoms.positions,
        nuclear_charges=atoms.charges + 1.0,
    )
    with pytest.raises(ValueError, match="agree exactly with batch.nuclear_charges"):
        ElectronNucleusPotential(atoms).local_energy(None, mismatched)


def test_electron_nucleus_potential_requires_agreement_with_batch_atomic_configuration() -> None:
    positions = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    atoms = _atoms([[0.0, 0.0]], [2.0])
    other_atoms = _atoms([[1.0, 0.0]], [2.0])
    batch = ElectronBatch(positions=positions, atomic_configuration=other_atoms)

    with pytest.raises(ValueError, match="agree exactly with batch.atomic_configuration"):
        ElectronNucleusPotential(atoms).local_energy(None, batch)


def test_electron_nucleus_potential_respects_batch_dtype_and_device() -> None:
    positions = torch.tensor([[[1.0], [2.0]]], dtype=torch.float32)
    atoms = _atoms([[0.0]], [2.0], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)

    result = ElectronNucleusPotential(atoms).local_energy(None, batch)

    assert result.total.dtype is torch.float32
    assert result.total.device == positions.device


@pytest.mark.parametrize(
    ("nuclei", "charges", "expected"),
    [
        ([[0.0, 0.0]], [2.0], [0.0]),
        ([[0.0, 0.0], [2.0, 0.0]], [2.0, 3.0], [3.0]),
        ([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]], [1.0, 2.0, 3.0], [2.0 / 3.0 + 3.0 / 4.0 + 6.0 / 5.0]),
    ],
)
def test_nucleus_nucleus_potential_exact_pair_sums_no_double_counting(nuclei, charges, expected) -> None:
    atoms = _atoms(nuclei, charges)
    batch = ElectronBatch(positions=torch.zeros(1, 1, 2, dtype=torch.float64))

    result = NucleusNucleusPotential(atoms).local_energy(None, batch)

    torch.testing.assert_close(result.total, torch.tensor(expected, dtype=torch.float64))
    torch.testing.assert_close(result.total, result.terms["nucleus_nucleus"])


def test_nucleus_nucleus_potential_broadcasts_across_batch() -> None:
    atoms = _atoms([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [2.0, 2.0])
    batch = ElectronBatch(positions=torch.zeros(3, 1, 3, dtype=torch.float64))

    result = NucleusNucleusPotential(atoms).local_energy(None, batch)

    torch.testing.assert_close(result.total, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))


def test_nucleus_nucleus_potential_requires_agreement_with_batch_context() -> None:
    atoms = _atoms([[0.0, 0.0], [2.0, 0.0]], [2.0, 3.0])
    batch = ElectronBatch(
        positions=torch.zeros(1, 1, 2, dtype=torch.float64),
        nuclear_positions=atoms.positions,
        nuclear_charges=atoms.charges + 1.0,
    )

    with pytest.raises(ValueError, match="agree exactly with batch.nuclear_charges"):
        NucleusNucleusPotential(atoms).local_energy(None, batch)


def test_nucleus_nucleus_potential_preserves_dtype_device_and_hamiltonian_aggregation() -> None:
    atoms = _atoms([[0.0, 0.0], [2.0, 0.0]], [2.0, 3.0], dtype=torch.float32)
    batch = ElectronBatch(positions=torch.zeros(2, 1, 2, dtype=torch.float32))

    result = local_energy({"nn": NucleusNucleusPotential(atoms)}, None, batch, return_terms=True)

    assert result.total.dtype == batch.positions.dtype
    assert result.total.device == batch.positions.device
    torch.testing.assert_close(result.terms["nn"], torch.tensor([3.0, 3.0], dtype=torch.float32))


def test_electron_nucleus_and_nucleus_nucleus_potential_compose_via_naive_evaluator() -> None:
    atoms = _atoms([[0.0, 0.0], [2.0, 0.0]], [1.0, 1.0])
    positions = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)

    result = local_energy(
        {"en": ElectronNucleusPotential(atoms), "nn": NucleusNucleusPotential(atoms)},
        None,
        batch,
        return_terms=True,
    )

    assert isinstance(result, LocalEnergyResult)
    torch.testing.assert_close(result.total, result.terms["en"] + result.terms["nn"])
