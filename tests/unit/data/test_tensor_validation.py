"""Tests for tensor-state validation hooks."""

from __future__ import annotations

import pytest
import torch
from typeguard import TypeCheckError

from tpen.data.partition import Partition
from tpen.data.real import (
    Feature,
    Interaction,
    Update,
    common_real_batch_size,
    common_real_dtype,
    common_real_particle_count,
    validate_matching_real_blocks,
    validate_real_update_geometry,
    zero_block,
)


def test_real_feature_requires_order_indexed_blocks_and_zero_channels() -> None:
    valid = Feature(
        [
            zero_block(batch_size=2, dtype=torch.float64),
            torch.zeros(2, 3, 4, dtype=torch.float64),
        ]
    )

    assert valid.validate() is valid

    with pytest.raises((TypeError, TypeCheckError), match="sequence"):
        Feature({1: torch.zeros(2, 3, 4, dtype=torch.float64)})
    with pytest.raises(ValueError, match="zero channels"):
        Feature([torch.zeros(2, 1, dtype=torch.float64)])


def test_zero_block_helper_centralizes_reserved_order_zero_layout() -> None:
    feature_zero = zero_block(batch_size=3, dtype=torch.float64)
    interaction_zero = zero_block(batch_size=3, paths=5, dtype=torch.float32)

    assert feature_zero.shape == (3, 0)
    assert feature_zero.dtype == torch.float64
    assert interaction_zero.shape == (3, 0, 5)
    assert interaction_zero.dtype == torch.float32

    with pytest.raises(ValueError, match="batch_size"):
        zero_block(batch_size=-1)
    with pytest.raises(ValueError, match="paths"):
        zero_block(paths=-1)


def test_real_tensor_validation_checks_batch_rank_and_particle_counts() -> None:
    with pytest.raises(ValueError, match="batch"):
        Update(
            [
                zero_block(batch_size=2, dtype=torch.float64),
                torch.zeros(3, 3, 4, dtype=torch.float64),
            ]
        )
    with pytest.raises(ValueError, match="dimensions"):
        Feature(
            [
                zero_block(batch_size=2, dtype=torch.float64),
                torch.zeros(2, 3, 4, 4, dtype=torch.float64),
            ]
        )
    with pytest.raises(ValueError, match="particle count"):
        Interaction(
            [
                zero_block(batch_size=2, paths=1, dtype=torch.float64),
                torch.zeros(2, 3, 1, 4, dtype=torch.float64),
                torch.zeros(2, 3, 1, 5, 5, dtype=torch.float64),
            ]
        )


def test_real_update_matching_validator_is_data_owned() -> None:
    feature = Feature(
        [
            zero_block(batch_size=2, dtype=torch.float64),
            torch.zeros(2, 3, 4, dtype=torch.float64),
        ]
    )
    update = Update(
        [
            zero_block(batch_size=2, dtype=torch.float64),
            torch.ones(2, 3, 4, dtype=torch.float64),
        ]
    )

    validate_matching_real_blocks(feature, update)

    with pytest.raises(ValueError, match="body-order"):
        validate_matching_real_blocks(
            feature,
            Update([zero_block(batch_size=2, dtype=torch.float64)]),
        )
    with pytest.raises(ValueError, match="Order-1"):
        validate_matching_real_blocks(
            feature,
            Update(
                [
                    zero_block(batch_size=2, dtype=torch.float64),
                    torch.ones(2, 3, 5, dtype=torch.float64),
                ]
            ),
        )


def test_real_update_geometry_validator_allows_channel_maps() -> None:
    feature = Feature(
        [
            zero_block(batch_size=2, dtype=torch.float64),
            torch.zeros(2, 3, 4, dtype=torch.float64),
        ]
    )
    update = Update(
        [
            zero_block(batch_size=2, dtype=torch.float64),
            torch.ones(2, 5, 4, dtype=torch.float64),
        ]
    )

    validate_real_update_geometry(feature, update)

    with pytest.raises(ValueError, match="tuple geometry"):
        validate_real_update_geometry(
            feature,
            Update(
                [
                    zero_block(batch_size=2, dtype=torch.float64),
                    torch.ones(2, 5, 5, dtype=torch.float64),
                ]
            ),
        )


def test_real_tensor_common_state_helpers_are_data_owned() -> None:
    feature = Feature(
        [
            zero_block(batch_size=2, dtype=torch.float64),
            torch.zeros(2, 3, 4, dtype=torch.float64),
        ]
    )
    update = Update(
        [
            zero_block(batch_size=2, dtype=torch.float64),
            torch.ones(2, 3, 4, dtype=torch.float64),
        ]
    )

    assert common_real_particle_count(feature, update) == 4
    assert common_real_batch_size(feature, update) == 2
    assert common_real_dtype(feature, update) is torch.float64

    with pytest.raises(ValueError, match="particle counts"):
        common_real_particle_count(
            feature,
            Update(
                [
                    zero_block(batch_size=2, dtype=torch.float64),
                    torch.ones(2, 3, 5, dtype=torch.float64),
                ]
            ),
        )
    with pytest.raises(ValueError, match="batch sizes"):
        common_real_batch_size(
            feature,
            Update(
                [
                    zero_block(batch_size=3, dtype=torch.float64),
                    torch.ones(3, 3, 4, dtype=torch.float64),
                ]
            ),
        )
    with pytest.raises(ValueError, match="dtypes"):
        common_real_dtype(
            feature,
            Update(
                [
                    zero_block(batch_size=2, dtype=torch.float32),
                    torch.ones(2, 3, 4, dtype=torch.float32),
                ]
            ),
        )


def test_partition_owns_activation_classification_and_module_keys() -> None:
    assert Partition((3,)).is_symmetric()
    assert Partition((1, 1, 1)).is_antisymmetric()
    assert not Partition((2, 1)).is_symmetric()
    assert not Partition((2, 1)).is_antisymmetric()
    assert Partition((2, 1)).key == "p2_1"
