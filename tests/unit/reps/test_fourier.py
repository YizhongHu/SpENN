"""Tests for scaffold Fourier transforms."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.irrep import IrrepFeature
from spenn.data.partition import Partition
from spenn.data.real import RealInteraction, zero_block
from spenn.nn.activation import GatedNormActivation
from spenn.reps import FourierTransform, InverseFourierTransform


def test_fourier_lifts_real_interactions_to_irrep_tail_dimensions() -> None:
    real = RealInteraction(
        [
            zero_block(paths=0, dtype=torch.float64),
            torch.arange(1 * 2 * 1 * 3, dtype=torch.float64).reshape(1, 2, 1, 3),
            torch.arange(1 * 2 * 1 * 3 * 3, dtype=torch.float64).reshape(1, 2, 1, 3, 3),
        ]
    )

    irrep = FourierTransform()(real)

    assert set(partition.parts for partition in irrep.blocks) == {(1,), (2,), (1, 1)}
    assert irrep[Partition((1,))].shape == (1, 2, 1, 3, 1, 1)
    assert irrep[Partition((2,))].shape == (1, 2, 1, 3, 3, 1, 1)
    torch.testing.assert_close(irrep[Partition((1,))][..., 0, 0], real.blocks[1])


def test_fourier_uses_slot_permutation_representations_for_order_two() -> None:
    pair = torch.tensor([[[[0.0, 2.0], [5.0, 0.0]]]], dtype=torch.float64)
    real = RealInteraction(
        [
            zero_block(paths=0, dtype=torch.float64),
            torch.empty(1, 0, 1, 2, dtype=torch.float64),
            pair.unsqueeze(2),
        ]
    )

    irrep = FourierTransform()(real)

    symmetric = irrep[Partition((2,))][..., 0, 0].squeeze(2)
    antisymmetric = irrep[Partition((1, 1))][..., 0, 0].squeeze(2)
    torch.testing.assert_close(symmetric, 0.5 * (pair + pair.transpose(-1, -2)))
    torch.testing.assert_close(antisymmetric, 0.5 * (pair - pair.transpose(-1, -2)))


def test_fourier_normalization_sets_activation_gate_scale() -> None:
    pair = torch.tensor([[[[0.0, 2.0], [5.0, 0.0]]]], dtype=torch.float64)
    real = RealInteraction(
        [
            zero_block(paths=0, dtype=torch.float64),
            torch.empty(1, 0, 1, 2, dtype=torch.float64),
            pair.unsqueeze(2),
        ]
    )
    partition = Partition((2,))

    irrep = FourierTransform(partitions=(partition,))(real)
    activated = GatedNormActivation(gate=nn.Identity())(irrep)

    normalized_block = irrep[partition]
    expected_real_projection = 0.5 * (pair + pair.transpose(-1, -2))
    torch.testing.assert_close(normalized_block[..., 0, 0].squeeze(2), expected_real_projection)
    torch.testing.assert_close(
        activated[partition],
        normalized_block * normalized_block.square().sum(dim=-2, keepdim=True),
    )


def test_inverse_fourier_recovers_path_aggregated_slot_fourier_transform() -> None:
    real = RealInteraction(
        [
            zero_block(paths=0, dtype=torch.float64),
            torch.arange(1 * 2 * 2 * 3, dtype=torch.float64).reshape(1, 2, 2, 3),
            torch.arange(1 * 2 * 2 * 3 * 3, dtype=torch.float64).reshape(1, 2, 2, 3, 3),
        ]
    )
    irrep_interaction = FourierTransform()(real)
    irrep_feature = IrrepFeature(
        {partition: tensor.sum(dim=2) for partition, tensor in irrep_interaction.items()}
    )

    update = InverseFourierTransform()(irrep_feature)

    torch.testing.assert_close(update.blocks[1], real.blocks[1].sum(dim=2))
    torch.testing.assert_close(update.blocks[2], real.blocks[2].sum(dim=2))


def test_order_three_inverse_fourier_roundtrip_uses_sage_cache() -> None:
    generator = torch.Generator().manual_seed(97531)
    real = RealInteraction(
        [
            zero_block(paths=0, dtype=torch.float64),
            torch.empty(1, 0, 1, 3, dtype=torch.float64),
            torch.empty(1, 0, 1, 3, 3, dtype=torch.float64),
            torch.randn(1, 2, 1, 3, 3, 3, generator=generator, dtype=torch.float64),
        ]
    )
    irrep_interaction = FourierTransform()(real)
    irrep_feature = IrrepFeature(
        {partition: tensor.sum(dim=2) for partition, tensor in irrep_interaction.items()}
    )

    update = InverseFourierTransform()(irrep_feature)

    torch.testing.assert_close(update.blocks[3], real.blocks[3].sum(dim=2))
