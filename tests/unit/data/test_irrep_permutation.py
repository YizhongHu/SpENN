"""Tests for irrep state particle permutation axes."""

from __future__ import annotations

import torch

from spenn.data.indices import permutation_index, permute_tuple_axes
from spenn.data.irrep import IrrepFeature, IrrepInteraction
from spenn.data.partition import Partition
from spenn.data.permutation import Permutation


def test_irrep_feature_permute_avoids_irrep_tail_axes_for_two_dimensional_irrep() -> None:
    partition = Partition((2, 1))
    tensor = torch.arange(1 * 1 * 3 * 3 * 3 * 2 * 2, dtype=torch.float64).reshape(1, 1, 3, 3, 3, 2, 2)
    state = IrrepFeature({partition: tensor})
    permutation = Permutation((2, 0, 1))

    permuted = state.permute(permutation)[partition]

    torch.testing.assert_close(
        permuted,
        permute_tuple_axes(tensor, permutation, axis_start=2, order=partition.order),
    )
    source = tuple(int(item) for item in permutation_index(permutation).tolist())
    torch.testing.assert_close(permuted[0, 0, 0, 1, 2], tensor[(0, 0, *source)])


def test_irrep_interaction_permute_avoids_irrep_tail_axes_for_two_dimensional_irrep() -> None:
    partition = Partition((2, 1))
    tensor = torch.arange(1 * 1 * 2 * 3 * 3 * 3 * 2 * 2, dtype=torch.float64).reshape(
        1,
        1,
        2,
        3,
        3,
        3,
        2,
        2,
    )
    state = IrrepInteraction({partition: tensor})
    permutation = Permutation((2, 0, 1))

    permuted = state.permute(permutation)[partition]

    torch.testing.assert_close(
        permuted,
        permute_tuple_axes(tensor, permutation, axis_start=3, order=partition.order),
    )
    source = tuple(int(item) for item in permutation_index(permutation).tolist())
    torch.testing.assert_close(permuted[0, 0, 1, 0, 1, 2], tensor[(0, 0, 1, *source)])
