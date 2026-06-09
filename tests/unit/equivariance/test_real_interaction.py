"""Tests for RealInteraction permutation actions."""

from __future__ import annotations

import torch

from spenn.data.indices import permute_tuple_axes
from spenn.data.permutation import Permutation
from spenn.data.real import RealInteraction, zero_block


def _interaction() -> RealInteraction:
    return RealInteraction(
        [
            zero_block(paths=4, dtype=torch.float64),
            torch.arange(1 * 2 * 4 * 3, dtype=torch.float64).reshape(1, 2, 4, 3),
            torch.arange(1 * 2 * 4 * 3 * 3, dtype=torch.float64).reshape(1, 2, 4, 3, 3),
        ]
    )


def test_real_interaction_identity_and_composition() -> None:
    interaction = _interaction()
    first = Permutation((1, 0, 2))
    second = Permutation((2, 1, 0))

    identity = interaction.permute(Permutation.identity(3))
    sequential = interaction.permute(first).permute(second)
    composed = interaction.permute(second.compose(first))

    torch.testing.assert_close(identity.blocks, interaction.blocks)
    torch.testing.assert_close(sequential.blocks, composed.blocks)


def test_real_interaction_preserves_path_axis() -> None:
    interaction = _interaction()
    permutation = Permutation((2, 0, 1))

    permuted = interaction.permute(permutation)

    assert permuted.blocks[1].shape[2] == interaction.blocks[1].shape[2]
    torch.testing.assert_close(
        permuted.blocks[1],
        permute_tuple_axes(interaction.blocks[1], permutation, axis_start=3, order=1),
    )
