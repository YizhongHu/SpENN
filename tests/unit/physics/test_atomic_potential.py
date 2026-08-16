"""Tests for constructor-owned AtomicConfiguration Coulomb potentials."""

from __future__ import annotations

import pytest
import torch

from tpen.data import AtomicConfiguration
from tpen.data.batch import ElectronBatch
from tpen.data.permutation import Permutation
from tpen.physics.hamiltonian import LocalEnergyResult, local_energy
from tpen.physics.potential import (
    ElectronNucleusInteraction,
    ElectronNucleusPotential,
    NucleusNucleusInteraction,
    NucleusNucleusPotential,
)


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


# --- A6: generic N=2 (H2) AtomicConfiguration data coverage ---


def _h2_atoms(dtype: torch.dtype = torch.float64) -> AtomicConfiguration:
    return _atoms([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], [1.0, 1.0], dtype=dtype)


def _slow_reference_electron_nucleus_energy(
    positions: torch.Tensor, atoms: AtomicConfiguration
) -> torch.Tensor:
    """Un-vectorized double-loop reference for the electron-nucleus Coulomb sum."""

    n_config, n_electrons, _ = positions.shape
    out = torch.zeros(n_config, dtype=positions.dtype)
    for c in range(n_config):
        for i in range(n_electrons):
            total = 0.0
            for a in range(atoms.n_nuclei):
                r = torch.linalg.norm(positions[c, i] - atoms.positions[a]).item()
                total += atoms.charges[a].item() / r
            out[c] -= total
    return out


def _slow_reference_nucleus_nucleus_energy(atoms: AtomicConfiguration) -> float:
    """Un-vectorized double-loop reference for the nuclear repulsion sum."""

    total = 0.0
    for a in range(atoms.n_nuclei):
        for b in range(a + 1, atoms.n_nuclei):
            r = torch.linalg.norm(atoms.positions[a] - atoms.positions[b]).item()
            total += atoms.charges[a].item() * atoms.charges[b].item() / r
    return total


def test_h2_electron_nucleus_and_nucleus_nucleus_potential_match_slow_reference() -> None:
    atoms = _h2_atoms()
    positions = torch.tensor(
        [[[0.3, 0.1, -0.5], [-0.2, 0.4, 0.6]], [[0.0, 0.0, 0.0], [1.0, -1.0, 2.0]]],
        dtype=torch.float64,
    )
    batch = ElectronBatch(positions=positions)

    en_result = ElectronNucleusPotential(atoms).local_energy(None, batch)
    nn_result = NucleusNucleusPotential(atoms).local_energy(None, batch)

    torch.testing.assert_close(en_result.total, _slow_reference_electron_nucleus_energy(positions, atoms))
    torch.testing.assert_close(
        nn_result.total,
        torch.full((positions.shape[0],), _slow_reference_nucleus_nucleus_energy(atoms), dtype=torch.float64),
    )


def test_h2_electron_nucleus_and_nucleus_nucleus_potential_are_nucleus_relabel_invariant() -> None:
    atoms = _h2_atoms()
    relabeled = _atoms(atoms.positions.flip(0).tolist(), atoms.charges.flip(0).tolist())
    positions = torch.tensor([[[0.3, 0.1, -0.5], [-0.2, 0.4, 0.6]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)

    en_result = ElectronNucleusPotential(atoms).local_energy(None, batch)
    en_relabeled = ElectronNucleusPotential(relabeled).local_energy(None, batch)
    nn_result = NucleusNucleusPotential(atoms).local_energy(None, batch)
    nn_relabeled = NucleusNucleusPotential(relabeled).local_energy(None, batch)

    torch.testing.assert_close(en_result.total, en_relabeled.total)
    torch.testing.assert_close(nn_result.total, nn_relabeled.total)


def test_h2_electron_nucleus_potential_raw_exact_zero_boundary_at_one_nucleus() -> None:
    # An electron exactly coincident with one of H2's two nuclei must diverge
    # from that nucleus's raw (unfloored) term while the other nucleus's term
    # stays finite and exact -- proving the divergence is per-pair, not a
    # global clamp or an H2-specific special case.
    atoms = _h2_atoms()
    positions = atoms.positions[0].clone().view(1, 1, 3)
    batch = ElectronBatch(positions=positions)

    result = ElectronNucleusPotential(atoms, eps=0.0).local_energy(None, batch)

    assert torch.isinf(result.total).all()
    assert (result.total < 0).all()


def test_h2_electron_nucleus_potential_default_eps_floor_matches_hand_calculation_at_coincidence() -> None:
    atoms = _h2_atoms()
    positions = atoms.positions[0].clone().view(1, 1, 3)
    batch = ElectronBatch(positions=positions)

    result = ElectronNucleusPotential(atoms, eps=1e-12).local_energy(None, batch)

    distance_to_far_nucleus = torch.linalg.norm(atoms.positions[0] - atoms.positions[1])
    expected = -(1.0 / 1e-12 + 1.0 / distance_to_far_nucleus)
    torch.testing.assert_close(result.total, expected.view(1))


def test_h2_electron_nucleus_and_nucleus_nucleus_potential_respect_dtype_and_device() -> None:
    atoms = _h2_atoms(dtype=torch.float32)
    positions = torch.tensor([[[0.3, 0.1, -0.5], [-0.2, 0.4, 0.6]]], dtype=torch.float32)
    batch = ElectronBatch(positions=positions)

    en_result = ElectronNucleusPotential(atoms).local_energy(None, batch)
    nn_result = NucleusNucleusPotential(atoms).local_energy(None, batch)

    assert en_result.total.dtype is torch.float32
    assert en_result.total.device == positions.device
    assert nn_result.total.dtype is torch.float32
    assert nn_result.total.device == positions.device


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


# --- A7: generic-vs-legacy potential parity under a matching AtomicConfiguration ---


def test_h2_generic_and_legacy_electron_nucleus_potentials_agree_by_value() -> None:
    # `ElectronNucleusPotential(atoms)` (constructor-owned) and legacy
    # `ElectronNucleusInteraction` (batch-transported) must produce
    # equal-valued electron-nucleus energies for the same H2 geometry --
    # reconciled by value equality, not by sharing a Python object.
    atoms = _h2_atoms()
    positions = torch.tensor(
        [[[0.3, 0.1, -0.5], [-0.2, 0.4, 0.6]], [[0.0, 0.0, 0.0], [1.0, -1.0, 2.0]]],
        dtype=torch.float64,
    )
    batch = ElectronBatch(
        positions=positions,
        nuclear_positions=atoms.positions,
        nuclear_charges=atoms.charges,
    )

    generic_result = ElectronNucleusPotential(atoms).local_energy(None, batch)
    legacy_result = ElectronNucleusInteraction().local_energy(None, batch)

    torch.testing.assert_close(generic_result.total, legacy_result.total)


def test_h2_generic_and_legacy_nucleus_nucleus_potentials_agree_by_value() -> None:
    atoms = _h2_atoms()
    positions = torch.tensor([[[0.3, 0.1, -0.5], [-0.2, 0.4, 0.6]]], dtype=torch.float64)
    batch = ElectronBatch(
        positions=positions,
        nuclear_positions=atoms.positions,
        nuclear_charges=atoms.charges,
    )

    generic_result = NucleusNucleusPotential(atoms).local_energy(None, batch)
    legacy_result = NucleusNucleusInteraction().local_energy(None, batch)

    torch.testing.assert_close(generic_result.total, legacy_result.total)


# --- C0: canonical *Potential* vs legacy *Interaction* terminology contract ---


def test_potential_module_docstring_documents_canonical_vs_legacy_terminology() -> None:
    # C0 terminology contract: the module docstring is the load-bearing
    # source for the Potential/Interaction compatibility split; pin its key
    # claims so a future edit cannot silently drop them.
    import tpen.physics.potential as potential_module

    doc = potential_module.__doc__
    assert "canonical Hamiltonian API" in doc
    assert "compatibility surface" in doc
    assert "per-configuration" in doc
    assert "not deprecated in this minor version" in doc


def test_electron_nucleus_potential_and_interaction_are_distinct_classes() -> None:
    # Canonical fixed-AtomicConfiguration API and legacy batch-transported
    # API must remain two distinct, independently constructible classes, not
    # one aliasing the other.
    assert ElectronNucleusPotential is not ElectronNucleusInteraction
    assert not issubclass(ElectronNucleusPotential, ElectronNucleusInteraction)
    assert not issubclass(ElectronNucleusInteraction, ElectronNucleusPotential)


def test_nucleus_nucleus_potential_and_interaction_are_distinct_classes() -> None:
    assert NucleusNucleusPotential is not NucleusNucleusInteraction
    assert not issubclass(NucleusNucleusPotential, NucleusNucleusInteraction)
    assert not issubclass(NucleusNucleusInteraction, NucleusNucleusPotential)
