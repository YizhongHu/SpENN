"""Tests for tensor-state validation hooks."""

from __future__ import annotations

import pytest
import torch
from typeguard import TypeCheckError

from spenn.data import IrrepFeature, IrrepInteraction, Partition, RealFeature, RealInteraction, RealUpdate, zero_block


def test_real_feature_requires_order_indexed_blocks_and_zero_channels() -> None:
    valid = RealFeature(
        [
            zero_block(batch_size=2, dtype=torch.float64),
            torch.zeros(2, 3, 4, dtype=torch.float64),
        ]
    )

    assert valid.validate() is valid

    with pytest.raises((TypeError, TypeCheckError), match="sequence"):
        RealFeature({1: torch.zeros(2, 3, 4, dtype=torch.float64)})
    with pytest.raises(ValueError, match="zero channels"):
        RealFeature([torch.zeros(2, 1, dtype=torch.float64)])


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
        RealUpdate(
            [
                zero_block(batch_size=2, dtype=torch.float64),
                torch.zeros(3, 3, 4, dtype=torch.float64),
            ]
        )
    with pytest.raises(ValueError, match="dimensions"):
        RealFeature(
            [
                zero_block(batch_size=2, dtype=torch.float64),
                torch.zeros(2, 3, 4, 4, dtype=torch.float64),
            ]
        )
    with pytest.raises(ValueError, match="particle count"):
        RealInteraction(
            [
                zero_block(batch_size=2, paths=1, dtype=torch.float64),
                torch.zeros(2, 3, 1, 4, dtype=torch.float64),
                torch.zeros(2, 3, 1, 5, 5, dtype=torch.float64),
            ]
        )


def test_irrep_feature_uses_partition_keys_and_validates_tail_dimensions() -> None:
    vector = Partition((2, 1))
    valid = IrrepFeature({vector: torch.zeros(2, 3, 4, 4, 4, 2, 2, dtype=torch.float64)})

    assert valid.validate() is valid
    assert valid[vector].shape[-2:] == (2, 2)

    with pytest.raises((TypeError, TypeCheckError), match="Partition|torch.Tensor"):
        IrrepFeature({1: {Partition((1,)): torch.zeros(2, 3, 4, 1, 1, dtype=torch.float64)}})
    with pytest.raises(ValueError, match="irrep dimensions"):
        IrrepFeature({vector: torch.zeros(2, 3, 4, 4, 4, 1, 1, dtype=torch.float64)})


def test_partition_owns_activation_classification_and_module_keys() -> None:
    assert Partition((3,)).is_symmetric()
    assert Partition((1, 1, 1)).is_antisymmetric()
    assert not Partition((2, 1)).is_symmetric()
    assert not Partition((2, 1)).is_antisymmetric()
    assert Partition((2, 1)).key == "p2_1"


def test_irrep_feature_checks_channels_for_same_order_but_interaction_is_looser() -> None:
    trivial = Partition((3,))
    vector = Partition((2, 1))

    with pytest.raises(ValueError, match="channel"):
        IrrepFeature(
            {
                trivial: torch.zeros(2, 2, 4, 4, 4, 1, 1, dtype=torch.float64),
                vector: torch.zeros(2, 3, 4, 4, 4, 2, 2, dtype=torch.float64),
            }
        )

    interaction = IrrepInteraction(
        {
            trivial: torch.zeros(2, 2, 5, 4, 4, 4, 1, 1, dtype=torch.float64),
            vector: torch.zeros(2, 3, 5, 4, 4, 4, 2, 2, dtype=torch.float64),
        }
    )
    assert interaction.validate() is interaction
