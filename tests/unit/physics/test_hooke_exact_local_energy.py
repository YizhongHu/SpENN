"""Unit tests for the Hamiltonian: exact Hooke local energies are constant.

These exercise ``local_energy`` over the kinetic + harmonic-trap +
electron-electron terms against the analytic Hooke eigenstates, asserting the
local energy is the exact eigenvalue everywhere (and has near-zero variance).
"""

from __future__ import annotations

import torch

from spenn.data.batch import ElectronBatch
from spenn.physics.hamiltonian import LocalEnergyResult, local_energy
from spenn.physics.hooke import HookeSingletExact, HookeTripletExact
from spenn.physics.kinetic import KineticEnergy
from spenn.physics.potential import ElectronElectronInteraction, HarmonicTrap

DTYPE = torch.float64
BATCH_SIZE = 64
ENERGY_ATOL = 1e-5
VARIANCE_ATOL = 1e-8


def _hooke_terms(wf) -> list:
    """Kinetic + harmonic trap + electron-electron terms for a Hooke pair."""
    return [KineticEnergy(), HarmonicTrap(omega=wf.omega), ElectronElectronInteraction()]


def _singlet_positions(seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    pos = torch.empty(BATCH_SIZE, 2, 3, dtype=DTYPE).normal_(generator=g) * 0.8
    # Displace electron 2 along x by 1.5 to guarantee r12 >= 1.5 > min_pair_distance
    pos[:, 1, 0] = pos[:, 0, 0] + 1.5
    return pos


def _triplet_positions(seed: int = 1) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    pos = torch.empty(BATCH_SIZE, 2, 3, dtype=DTYPE).normal_(generator=g) * 0.8
    # Displace electron 2 along x by 1.5 to guarantee r12 >= 1.5 > min_pair_distance
    pos[:, 1, 0] = pos[:, 0, 0] + 1.5
    # Set z2 = z1 + 0.5 to guarantee |z1 - z2| = 0.5 > min_triplet_node_distance
    pos[:, 1, 2] = pos[:, 0, 2] + 0.5
    return pos


def test_singlet_local_energy_constant_at_exact_energy() -> None:
    wf = HookeSingletExact()
    terms = _hooke_terms(wf)
    positions = _singlet_positions()
    batch = ElectronBatch(positions=positions)

    result = local_energy(terms, wf, batch, return_terms=True)
    assert isinstance(result, LocalEnergyResult)
    eloc = result.total

    assert eloc.shape == (BATCH_SIZE,)
    assert torch.all(torch.isfinite(eloc))
    assert torch.allclose(eloc, torch.full_like(eloc, wf.exact_energy), atol=ENERGY_ATOL)
    assert eloc.std().item() < VARIANCE_ATOL**0.5


def test_singlet_local_energy_variance_near_zero() -> None:
    wf = HookeSingletExact()
    terms = _hooke_terms(wf)
    batch = ElectronBatch(positions=_singlet_positions())

    eloc = local_energy(terms, wf, batch)
    assert isinstance(eloc, torch.Tensor)
    assert eloc.var().item() < VARIANCE_ATOL


def test_triplet_local_energy_constant_at_exact_energy() -> None:
    wf = HookeTripletExact()
    terms = _hooke_terms(wf)
    positions = _triplet_positions()
    batch = ElectronBatch(positions=positions)

    result = local_energy(terms, wf, batch, return_terms=True)
    assert isinstance(result, LocalEnergyResult)
    eloc = result.total

    assert eloc.shape == (BATCH_SIZE,)
    assert torch.all(torch.isfinite(eloc))
    assert torch.allclose(eloc, torch.full_like(eloc, wf.exact_energy), atol=ENERGY_ATOL)
    assert eloc.std().item() < VARIANCE_ATOL**0.5


def test_triplet_local_energy_variance_near_zero() -> None:
    wf = HookeTripletExact()
    terms = _hooke_terms(wf)
    batch = ElectronBatch(positions=_triplet_positions())

    eloc = local_energy(terms, wf, batch)
    assert isinstance(eloc, torch.Tensor)
    assert eloc.var().item() < VARIANCE_ATOL


def test_local_energy_term_decomposition_sums_to_total() -> None:
    wf = HookeSingletExact()
    terms = _hooke_terms(wf)
    batch = ElectronBatch(positions=_singlet_positions())

    result = local_energy(terms, wf, batch, return_terms=True)
    assert isinstance(result, LocalEnergyResult)
    assert set(result.terms.keys()) == {"kinetic", "harmonic_trap", "electron_electron"}

    reconstructed = sum(result.terms.values())
    assert torch.allclose(result.total, reconstructed)
