"""Contract tests for versioned Hooke orbital-basis dispatch."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from spenn.config import basis_feature_dim
from spenn.data.batch import ElectronBatch
from spenn.nn import HookeOrbitalBasis
from spenn.nn.basis import ElectronBasisFeatures
from tests.helpers.equivariance import assert_equivariant_all


_ORDER_V1 = "shell-major-coordinate-priority-v1"
_TOTAL_SHELL_D2_S2 = (
    (0, 0),
    (1, 0),
    (0, 1),
    (2, 0),
    (1, 1),
    (0, 2),
)
_TOTAL_SHELL_D3_S2 = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (2, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (0, 2, 0),
    (0, 1, 1),
    (0, 0, 2),
)
_BOX_D3_K2 = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)


def _hermite(order: int, xi: torch.Tensor) -> torch.Tensor:
    """Return the physicists' Hermite polynomial without production helpers."""

    previous = torch.ones_like(xi)
    if order == 0:
        return previous
    current = 2.0 * xi
    for index in range(1, order):
        previous, current = current, 2.0 * xi * current - 2.0 * index * previous
    return current


def _legacy_axiswise_reference(
    positions: torch.Tensor,
    *,
    omega: float,
    max_shell: int,
    include_gaussian_factor: bool,
) -> torch.Tensor:
    """Freeze V1 coordinate-major, order-fastest axiswise feature semantics."""

    xi = positions * omega**0.5
    channels = []
    for coordinate in range(positions.shape[-1]):
        gaussian = torch.exp(-0.5 * xi[..., coordinate].square())
        for order in range(max_shell + 1):
            value = _hermite(order, xi[..., coordinate])
            channels.append(value * gaussian if include_gaussian_factor else value)
    return torch.stack(channels, dim=-1)


def _product_reference(
    positions: torch.Tensor,
    *,
    omega: float,
    multi_indices: tuple[tuple[int, ...], ...],
    include_gaussian_factor: bool,
) -> torch.Tensor:
    """Evaluate product orbitals from a test-local Hermite recurrence."""

    xi = positions * omega**0.5
    channels = []
    for multi_index in multi_indices:
        value = torch.ones_like(xi[..., 0])
        for coordinate, order in enumerate(multi_index):
            value = value * _hermite(order, xi[..., coordinate])
        channels.append(value)
    features = torch.stack(channels, dim=-1)
    if include_gaussian_factor:
        features = features * torch.exp(-0.5 * xi.square().sum(dim=-1, keepdim=True))
    return features


def _product_basis(
    *,
    spatial_dim: int,
    truncation: str,
    bound: int,
    include_gaussian_factor: bool,
    include_spin: bool = False,
) -> HookeOrbitalBasis:
    """Build an explicit V2 product basis with only its relevant bound."""

    bound_kwargs = {"max_total_shell": bound} if truncation == "total_shell" else {"box_size": bound}
    return HookeOrbitalBasis(
        omega=0.5,
        spatial_dim=spatial_dim,
        basis_semantics="product_v2",
        truncation=truncation,
        include_gaussian_factor=include_gaussian_factor,
        include_spin=include_spin,
        **bound_kwargs,
    )


def test_legacy_axiswise_v1_dispatch_preserves_frozen_output_contract() -> None:
    """Old unversioned configs remain V1 rather than silently becoming products."""

    positions = torch.tensor(
        [[[0.25, -0.75, 1.25], [-1.0, 0.5, 0.125]]], dtype=torch.float64
    )
    spins = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions, spins=spins)

    implicit_legacy = HookeOrbitalBasis(omega=0.5, max_shell=2, spatial_dim=3)
    explicit_legacy = HookeOrbitalBasis(
        omega=0.5,
        max_shell=2,
        spatial_dim=3,
        basis_semantics="axiswise_v1",
    )

    expected = torch.cat(
        [
            _legacy_axiswise_reference(
                positions,
                omega=0.5,
                max_shell=2,
                include_gaussian_factor=True,
            ),
            spins.unsqueeze(-1),
        ],
        dim=-1,
    )
    implicit_features = implicit_legacy(batch)
    explicit_features = explicit_legacy(batch)

    assert implicit_legacy.out_features == 10
    assert implicit_features.metadata["basis_semantics"] == "axiswise_v1"
    assert implicit_features.pair is None
    torch.testing.assert_close(implicit_features.one_body, expected)
    torch.testing.assert_close(explicit_features.one_body, expected)


@pytest.mark.parametrize(
    ("truncation", "bound", "expected_indices"),
    [
        ("total_shell", 2, _TOTAL_SHELL_D3_S2),
        ("cartesian_box", 2, _BOX_D3_K2),
    ],
)
def test_product_v2_exposes_canonical_immutable_multi_index_order(
    truncation: str,
    bound: int,
    expected_indices: tuple[tuple[int, ...], ...],
) -> None:
    """Both truncations use one shell-major, coordinate-priority order."""

    basis = _product_basis(
        spatial_dim=3,
        truncation=truncation,
        bound=bound,
        include_gaussian_factor=False,
        include_spin=True,
    )

    assert basis.multi_indices == expected_indices
    assert isinstance(basis.multi_indices, tuple)
    assert all(isinstance(index, tuple) for index in basis.multi_indices)
    assert len(basis.multi_indices) == len(set(basis.multi_indices))
    assert basis.coordinate_features == len(expected_indices)
    assert basis.out_features == len(expected_indices) + 1


@pytest.mark.parametrize(
    ("truncation", "bound"),
    [("total_shell", 2), ("cartesian_box", 3)],
)
def test_product_v2_one_dimension_reduces_to_hermite_sequence(
    truncation: str,
    bound: int,
) -> None:
    """One-dimensional product modes agree with the independent Hermite oracle."""

    positions = torch.tensor([[[-0.4], [0.3], [1.2]]], dtype=torch.float64)
    expected_indices = ((0,), (1,), (2,))
    basis = _product_basis(
        spatial_dim=1,
        truncation=truncation,
        bound=bound,
        include_gaussian_factor=False,
    )

    assert basis.multi_indices == expected_indices
    torch.testing.assert_close(
        basis(ElectronBatch(positions=positions)).one_body,
        _product_reference(
            positions,
            omega=0.5,
            multi_indices=expected_indices,
            include_gaussian_factor=False,
        ),
    )


def test_product_v2_total_shell_matches_independent_product_oracle_and_metadata() -> None:
    """V2 channels are true multidimensional products, including mixed terms."""

    positions = torch.tensor(
        [
            [[0.25, -0.75, 1.25], [-1.0, 0.5, 0.125]],
            [[0.6, -0.2, 0.9], [1.1, 0.4, -0.3]],
        ],
        dtype=torch.float64,
    )
    batch = ElectronBatch(positions=positions)
    basis = _product_basis(
        spatial_dim=3,
        truncation="total_shell",
        bound=2,
        include_gaussian_factor=False,
    )

    features = basis(batch)
    expected = _product_reference(
        positions,
        omega=0.5,
        multi_indices=_TOTAL_SHELL_D3_S2,
        include_gaussian_factor=False,
    )

    assert isinstance(features, ElectronBasisFeatures)
    assert features.one_body.shape == (*positions.shape[:-1], len(_TOTAL_SHELL_D3_S2))
    assert features.one_body.dtype is torch.float64
    assert features.pair is None
    assert features.metadata["basis_semantics"] == "product_v2"
    assert features.metadata["truncation"] == "total_shell"
    assert features.metadata["truncation_bound"] == 2
    assert features.metadata["multi_index_order"] == _ORDER_V1
    assert features.metadata["include_gaussian_factor"] is False
    assert "multi_indices" not in features.metadata
    torch.testing.assert_close(features.one_body, expected)


def test_product_v2_d2_sum_order_preserves_origin_parity_and_mixed_channel() -> None:
    """Two-dimensional sum-order channels cover parity and the ``(1, 1)`` term."""

    positions = torch.tensor([[[0.0, 0.0], [0.25, -0.75]]], dtype=torch.float64)
    basis = _product_basis(
        spatial_dim=2,
        truncation="total_shell",
        bound=2,
        include_gaussian_factor=False,
    )

    features = basis(ElectronBatch(positions=positions))
    expected = _product_reference(
        positions,
        omega=0.5,
        multi_indices=_TOTAL_SHELL_D2_S2,
        include_gaussian_factor=False,
    )

    torch.testing.assert_close(features.one_body, expected)
    torch.testing.assert_close(
        features.one_body[0, 0],
        torch.tensor([1.0, 0.0, 0.0, -2.0, 0.0, -2.0], dtype=torch.float64),
    )


def test_product_v2_cartesian_box_matches_independent_product_oracle() -> None:
    """Cartesian-box truncation changes index admission, not product evaluation."""

    positions = torch.tensor([[[0.25, -0.75, 1.25]]], dtype=torch.float64)
    basis = _product_basis(
        spatial_dim=3,
        truncation="cartesian_box",
        bound=2,
        include_gaussian_factor=False,
    )

    torch.testing.assert_close(
        basis(ElectronBatch(positions=positions)).one_body,
        _product_reference(
            positions,
            omega=0.5,
            multi_indices=_BOX_D3_K2,
            include_gaussian_factor=False,
        ),
    )


def test_product_v2_gaussian_is_single_radial_factor_for_every_channel() -> None:
    """Input Gaussian multiplies each product channel once, not once per axis."""

    positions = torch.tensor([[[0.25, -0.75, 1.25]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    polynomial = _product_basis(
        spatial_dim=3,
        truncation="total_shell",
        bound=2,
        include_gaussian_factor=False,
    )
    orbital = _product_basis(
        spatial_dim=3,
        truncation="total_shell",
        bound=2,
        include_gaussian_factor=True,
    )

    gaussian = torch.exp(-0.5 * 0.5 * positions.square().sum(dim=-1, keepdim=True))
    torch.testing.assert_close(orbital(batch).one_body, polynomial(batch).one_body * gaussian)


def test_product_v2_preserves_spin_sample_axes_and_particle_equivariance() -> None:
    """Product features retain ElectronBasis typed-output and permutation contracts."""

    generator = torch.Generator().manual_seed(7)
    positions = torch.randn(2, 3, 4, 3, generator=generator, dtype=torch.float64)
    spins = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64).repeat(2, 3, 1)
    batch = ElectronBatch(positions=positions, spins=spins)
    basis = _product_basis(
        spatial_dim=3,
        truncation="cartesian_box",
        bound=2,
        include_gaussian_factor=False,
        include_spin=True,
    ).to(dtype=torch.float64)

    features = basis(batch)

    assert features.one_body.shape == (2, 3, 4, len(_BOX_D3_K2) + 1)
    torch.testing.assert_close(features.one_body[..., -1], spins)
    assert_equivariant_all(basis, batch)


def test_product_v2_mixed_channel_gradient_matches_independent_oracle() -> None:
    """Mixed product channel remains differentiable for local-energy use."""

    positions = torch.tensor([[[0.25, -0.75, 1.25]]], dtype=torch.float64, requires_grad=True)
    basis = _product_basis(
        spatial_dim=3,
        truncation="total_shell",
        bound=2,
        include_gaussian_factor=False,
    ).to(dtype=torch.float64)

    actual = basis(ElectronBatch(positions=positions)).one_body[..., _TOTAL_SHELL_D3_S2.index((1, 1, 0))].sum()
    actual_gradient = torch.autograd.grad(actual, positions)[0]

    reference_positions = positions.detach().clone().requires_grad_(True)
    expected = _product_reference(
        reference_positions,
        omega=0.5,
        multi_indices=_TOTAL_SHELL_D3_S2,
        include_gaussian_factor=False,
    )[..., _TOTAL_SHELL_D3_S2.index((1, 1, 0))].sum()
    expected_gradient = torch.autograd.grad(expected, reference_positions)[0]

    assert torch.isfinite(actual_gradient).all()
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "basis_semantics": "product_v2",
            "truncation": "total_shell",
            "max_total_shell": 1,
        },
        {
            "basis_semantics": "product_v2",
            "max_total_shell": 1,
            "include_gaussian_factor": False,
        },
        {
            "basis_semantics": "product_v2",
            "truncation": "total_shell",
            "include_gaussian_factor": False,
        },
        {
            "basis_semantics": "product_v2",
            "truncation": "total_shell",
            "max_total_shell": 1,
            "box_size": 2,
            "include_gaussian_factor": False,
        },
        {
            "basis_semantics": "product_v2",
            "truncation": "cartesian_box",
            "max_total_shell": 1,
            "include_gaussian_factor": False,
        },
        {
            "basis_semantics": "product_v2",
            "truncation": "total_shell",
            "max_total_shell": 1,
            "max_shell": 1,
            "include_gaussian_factor": False,
        },
        {
            "basis_semantics": "product_v2",
            "truncation": "not-a-truncation",
            "max_total_shell": 1,
            "include_gaussian_factor": False,
        },
        {
            "basis_semantics": "product_v2",
            "truncation": "total_shell",
            "max_total_shell": -1,
            "include_gaussian_factor": False,
        },
        {
            "basis_semantics": "product_v2",
            "truncation": "cartesian_box",
            "box_size": 0,
            "include_gaussian_factor": False,
        },
        {
            "basis_semantics": "unknown_v9",
            "max_shell": 1,
            "include_gaussian_factor": True,
        },
        {
            "basis_semantics": "axiswise_v1",
            "max_shell": 1,
            "truncation": "total_shell",
            "max_total_shell": 1,
            "include_gaussian_factor": True,
        },
        {
            "truncation": "total_shell",
            "max_total_shell": 1,
            "include_gaussian_factor": False,
        },
    ],
)
def test_versioned_dispatcher_rejects_ambiguous_or_mixed_arguments(kwargs: dict[str, object]) -> None:
    """Old and product contracts must not silently share an unversioned shape."""

    with pytest.raises(ValueError):
        HookeOrbitalBasis(omega=0.5, spatial_dim=2, **kwargs)


@pytest.mark.parametrize(
    ("truncation", "bound", "expected_width"),
    [("total_shell", 2, 10), ("cartesian_box", 2, 8)],
)
def test_product_v2_config_resolver_uses_selected_truncation_width(
    truncation: str,
    bound: int,
    expected_width: int,
) -> None:
    """Resolved configs size downstream embedding from V2's actual width."""

    bound_config = {"max_total_shell": bound} if truncation == "total_shell" else {"box_size": bound}
    config = OmegaConf.create(
        {
            "_target_": "spenn.nn.HookeOrbitalBasis",
            "omega": 0.5,
            "spatial_dim": 3,
            "basis_semantics": "product_v2",
            "truncation": truncation,
            "include_gaussian_factor": False,
            "include_spin": False,
            **bound_config,
        }
    )

    assert basis_feature_dim(config) == expected_width
