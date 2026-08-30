"""Tests for the immutable `AtomicConfiguration` nuclear geometry value."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from tpen.data import AtomicConfiguration
from tpen.data.atomic_configuration import strict_equal_atomic_configurations


def _helium() -> AtomicConfiguration:
    return AtomicConfiguration(positions=torch.zeros(1, 3), charges=torch.tensor([2.0]))


def _hydrogen_molecule() -> AtomicConfiguration:
    positions = torch.tensor([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    charges = torch.tensor([1.0, 1.0])
    return AtomicConfiguration(positions=positions, charges=charges)


def test_helium_configuration_reports_shapes() -> None:
    config = _helium()

    assert config.n_nuclei == 1
    assert config.spatial_dim == 3
    assert config.validate() is config


def test_hydrogen_molecule_configuration_reports_shapes() -> None:
    config = _hydrogen_molecule()

    assert config.n_nuclei == 2
    assert config.spatial_dim == 3
    assert config.validate() is config


def test_rejects_wrong_rank_positions() -> None:
    with pytest.raises(ValueError, match="n_nuclei, spatial_dim"):
        AtomicConfiguration(positions=torch.zeros(3), charges=torch.tensor([1.0]))


def test_rejects_mismatched_charge_count() -> None:
    with pytest.raises(ValueError, match="charges must have shape"):
        AtomicConfiguration(positions=torch.zeros(2, 3), charges=torch.tensor([1.0]))


def test_rejects_nonpositive_charges() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        AtomicConfiguration(positions=torch.zeros(1, 3), charges=torch.tensor([0.0]))
    with pytest.raises(ValueError, match="strictly positive"):
        AtomicConfiguration(positions=torch.zeros(1, 3), charges=torch.tensor([-1.0]))


def test_rejects_nonfinite_positions_and_charges() -> None:
    with pytest.raises(ValueError, match="positions must be finite"):
        AtomicConfiguration(positions=torch.tensor([[float("nan"), 0.0, 0.0]]), charges=torch.tensor([1.0]))
    with pytest.raises(ValueError, match="charges must be finite"):
        AtomicConfiguration(positions=torch.zeros(1, 3), charges=torch.tensor([float("inf")]))


def test_rejects_colliding_nuclei() -> None:
    with pytest.raises(ValueError, match="distinct positions"):
        AtomicConfiguration(positions=torch.zeros(2, 3), charges=torch.tensor([1.0, 1.0]))


def test_caller_mutation_of_source_tensors_does_not_leak_in() -> None:
    positions = torch.zeros(1, 3)
    charges = torch.tensor([2.0])
    config = AtomicConfiguration(positions=positions, charges=charges)

    positions[0, 0] = 99.0
    charges[0] = 99.0

    assert config.positions[0, 0].item() == 0.0
    assert config.charges[0].item() == 2.0


def test_is_frozen_dataclass() -> None:
    config = _helium()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.positions = torch.zeros(1, 3)  # type: ignore[misc]


def test_to_materializes_dtype() -> None:
    config = _helium()

    moved = config.to(dtype=torch.float64)

    assert moved.dtype == torch.float64
    assert moved.positions.dtype == torch.float64
    assert moved.charges.dtype == torch.float64
    assert config.dtype == torch.float32


def test_hydrogen_molecule_to_materializes_dtype_and_preserves_n_nuclei() -> None:
    config = _hydrogen_molecule()

    moved = config.to(dtype=torch.float64)

    assert moved.dtype == torch.float64
    assert moved.n_nuclei == 2
    assert moved.positions.dtype == torch.float64
    assert moved.charges.dtype == torch.float64
    assert config.dtype == torch.float32


def test_hydrogen_molecule_nucleus_relabel_produces_a_distinct_but_valid_configuration() -> None:
    # Relabeling (permuting) the nucleus axis is ordinary data reordering, not
    # a physically different molecule; both orderings must independently
    # validate and compare unequal only because AtomicConfiguration equality
    # is positional, not set-based.
    config = _hydrogen_molecule()
    relabeled = AtomicConfiguration(
        positions=config.positions.flip(0),
        charges=config.charges.flip(0),
    )

    assert relabeled.validate() is relabeled
    assert relabeled.n_nuclei == config.n_nuclei
    torch.testing.assert_close(relabeled.positions, config.positions.flip(0))


def test_compare_detects_close_and_far_configurations() -> None:
    config = _helium()
    close = AtomicConfiguration(positions=torch.zeros(1, 3) + 1e-9, charges=torch.tensor([2.0]))
    far = AtomicConfiguration(positions=torch.ones(1, 3), charges=torch.tensor([2.0]))

    is_close, metrics = config.compare(close)
    assert is_close
    assert metrics["positions_max_abs_error"] < 1e-6

    is_close, metrics = config.compare(far)
    assert not is_close
    assert metrics["positions_max_abs_error"] > 0.5


def test_compare_mismatched_shapes_reports_infinite_error() -> None:
    config = _helium()
    other = _hydrogen_molecule()

    is_close, metrics = config.compare(other)

    assert not is_close
    assert metrics["positions_max_abs_error"] == float("inf")


def test_equality_and_hash_are_value_based() -> None:
    a = _helium()
    b = AtomicConfiguration(positions=torch.zeros(1, 3), charges=torch.tensor([2.0]))
    c = AtomicConfiguration(positions=torch.zeros(1, 3), charges=torch.tensor([3.0]))

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != "not a configuration"


def test_content_id_is_reproducible_and_device_dtype_independent() -> None:
    config = _helium()
    moved = config.to(dtype=torch.float64)

    assert config.content_id() == moved.content_id()

    different = _hydrogen_molecule()
    assert config.content_id() != different.content_id()


def test_strict_geometry_comparison_is_symmetric_and_promotes_without_narrowing() -> None:
    base_positions = torch.tensor([[0.25, 0.5, 0.75]], dtype=torch.float32)
    base_charges = torch.tensor([1.5], dtype=torch.float32)
    cases = [
        (
            AtomicConfiguration(base_positions, base_charges),
            AtomicConfiguration(
                base_positions.to(torch.float64) + 2**-25,
                base_charges.to(torch.float64),
            ),
        ),
        (
            AtomicConfiguration(base_positions, base_charges),
            AtomicConfiguration(
                base_positions.to(torch.float64),
                base_charges.to(torch.float64) + 2**-25,
            ),
        ),
        (
            AtomicConfiguration(
                torch.tensor([[0, 1, 2]], dtype=torch.int64),
                torch.tensor([2], dtype=torch.int64),
            ),
            AtomicConfiguration(
                torch.tensor([[0.25, 1.0, 2.0]], dtype=torch.float64),
                torch.tensor([2.0], dtype=torch.float64),
            ),
        ),
    ]

    for left, right in cases:
        assert not strict_equal_atomic_configurations(left, right)
        assert not strict_equal_atomic_configurations(right, left)

    exact_left = AtomicConfiguration(base_positions, torch.tensor([1.0], dtype=torch.float32))
    exact_right = AtomicConfiguration(
        base_positions.to(torch.float64), torch.tensor([1.0], dtype=torch.float64)
    )
    assert strict_equal_atomic_configurations(exact_left, exact_right)
    assert strict_equal_atomic_configurations(exact_right, exact_left)
