"""Unit tests for model-side electron bases (PR8.8)."""

from __future__ import annotations

import pytest
import torch

from spenn.data.batch import ElectronBatch
from spenn.nn import HookeHermiteBasis, HookeOrbitalBasis, RawCoordinateBasis
from spenn.nn.basis import ElectronBasisFeatures
from spenn.trace import Trace
from tests.helpers.equivariance import assert_equivariant_all


def _batch(n_electrons: int = 4, n_walkers: int = 3) -> ElectronBatch:
    """Return a tiny spin-resolved electron batch for basis tests."""

    generator = torch.Generator().manual_seed(2024)
    positions = torch.randn(n_walkers, n_electrons, 3, generator=generator, dtype=torch.float64)
    pattern = torch.tensor([1.0 if index % 2 == 0 else -1.0 for index in range(n_electrons)], dtype=torch.float64)
    spins = pattern.unsqueeze(0).repeat(n_walkers, 1)
    return ElectronBatch(positions=positions, spins=spins)


def test_raw_coordinate_basis_is_particle_equivariant() -> None:
    basis = RawCoordinateBasis(spatial_dim=3).to(dtype=torch.float64)
    assert_equivariant_all(basis, _batch())


def test_hooke_hermite_basis_is_particle_equivariant() -> None:
    basis = HookeHermiteBasis(omega=0.5, max_order=3, spatial_dim=3).to(dtype=torch.float64)
    assert_equivariant_all(basis, _batch())


def test_hooke_orbital_basis_is_particle_equivariant() -> None:
    basis = HookeOrbitalBasis(omega=0.5, max_shell=2, spatial_dim=3).to(dtype=torch.float64)
    assert_equivariant_all(basis, _batch())


def test_basis_output_is_typed_features_not_batch() -> None:
    batch = _batch()
    for basis in (
        RawCoordinateBasis(spatial_dim=3),
        HookeHermiteBasis(omega=0.5, max_order=2, spatial_dim=3),
        HookeOrbitalBasis(omega=0.5, max_shell=2, spatial_dim=3),
    ):
        features = basis(batch)
        assert isinstance(features, ElectronBasisFeatures)
        assert not isinstance(features, ElectronBatch)
        assert features.one_body.shape[:-1] == batch.positions.shape[:-1]
        assert features.n_electrons == batch.n_electrons


def test_basis_records_features_to_trace() -> None:
    batch = _batch()
    basis = RawCoordinateBasis(spatial_dim=3, trace_name="basis")

    with Trace.capture(model=basis) as trace:
        basis(batch)

    assert any(
        entry.slot == "features" and isinstance(entry.value, ElectronBasisFeatures)
        for entry in trace
    )
    assert "basis/output" in trace.keys()
    assert isinstance(trace["basis/output"].value, ElectronBasisFeatures)


@pytest.mark.parametrize("include_spin", [True, False])
@pytest.mark.parametrize(
    ("max_order", "max_shell"),
    [(2, 1), (3, 2), (4, 3)],
)
def test_basis_output_shapes_match_order_or_shell(
    max_order: int, max_shell: int, include_spin: bool
) -> None:
    batch = _batch()
    spin_channels = 1 if include_spin else 0
    spatial_dim = 3

    raw = RawCoordinateBasis(spatial_dim=spatial_dim, include_spin=include_spin)
    assert raw.out_features == spatial_dim + spin_channels
    assert raw(batch).one_body.shape[-1] == raw.out_features

    hermite = HookeHermiteBasis(
        omega=0.5, max_order=max_order, spatial_dim=spatial_dim, include_spin=include_spin
    )
    assert hermite.out_features == spatial_dim * (max_order + 1) + spin_channels
    assert hermite(batch).one_body.shape[-1] == hermite.out_features

    orbital = HookeOrbitalBasis(
        omega=0.5, max_shell=max_shell, spatial_dim=spatial_dim, include_spin=include_spin
    )
    assert orbital.out_features == spatial_dim * (max_shell + 1) + spin_channels
    assert orbital(batch).one_body.shape[-1] == orbital.out_features


def test_hermite_and_orbital_differ_only_by_gaussian_factor() -> None:
    batch = _batch()
    hermite = HookeHermiteBasis(omega=0.5, max_order=2, spatial_dim=3, include_spin=False)
    orbital = HookeOrbitalBasis(
        omega=0.5, max_shell=2, spatial_dim=3, include_spin=False, include_gaussian_factor=True
    )
    no_gaussian = HookeOrbitalBasis(
        omega=0.5, max_shell=2, spatial_dim=3, include_spin=False, include_gaussian_factor=False
    )
    # With the same scale and no Gaussian factor, the orbital basis reduces to
    # the Hermite polynomials; enabling the factor strictly changes the output.
    torch.testing.assert_close(hermite(batch).one_body, no_gaussian(batch).one_body)
    assert not torch.allclose(orbital(batch).one_body, no_gaussian(batch).one_body)


def test_raw_basis_reproduces_coordinate_spin_vector() -> None:
    batch = _batch()
    features = RawCoordinateBasis(spatial_dim=3, include_spin=True)(batch)
    expected = torch.cat([batch.positions, batch.spins.unsqueeze(-1)], dim=-1)
    torch.testing.assert_close(features.one_body, expected)
