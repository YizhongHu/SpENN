"""Tests for RealFeature permutation actions."""

from __future__ import annotations

from itertools import permutations

import torch

from spenn.data import Permutation, RealFeature, zero_block


def _feature() -> RealFeature:
    return RealFeature(
        [
            zero_block(dtype=torch.float64),
            torch.arange(1 * 2 * 3, dtype=torch.float64).reshape(1, 2, 3),
            torch.arange(1 * 2 * 3 * 3, dtype=torch.float64).reshape(1, 2, 3, 3),
        ]
    )


def test_real_feature_identity_and_composition() -> None:
    feature = _feature()
    first = Permutation((1, 0, 2))
    second = Permutation((2, 1, 0))

    identity = feature.permute(Permutation.identity(3))
    sequential = feature.permute(first).permute(second)
    composed = feature.permute(second.compose(first))

    torch.testing.assert_close(identity.blocks, feature.blocks)
    torch.testing.assert_close(sequential.blocks, composed.blocks)


def test_real_feature_permute_matches_active_axis_indexing() -> None:
    feature = _feature()
    permutation = Permutation((2, 0, 1))

    permuted = feature.permute(permutation)

    index = torch.tensor(permutation.inverse().image)
    torch.testing.assert_close(permuted.blocks[1], feature.blocks[1].index_select(2, index))
    torch.testing.assert_close(permuted.blocks[2], feature.blocks[2].index_select(2, index).index_select(3, index))


def test_real_feature_permute_all_small_permutations_and_orders() -> None:
    for n_particles in range(1, 6):
        feature = RealFeature(
            [
                zero_block(dtype=torch.float64),
                torch.arange(2 * n_particles, dtype=torch.float64).reshape(1, 2, n_particles),
                torch.arange(2 * n_particles**2, dtype=torch.float64).reshape(1, 2, n_particles, n_particles),
                torch.arange(2 * n_particles**3, dtype=torch.float64).reshape(
                    1,
                    2,
                    n_particles,
                    n_particles,
                    n_particles,
                ),
            ]
        )
        for image in permutations(range(n_particles)):
            permutation = Permutation(tuple(image))
            permuted = feature.permute(permutation)
            index = torch.tensor(permutation.inverse().image)
            torch.testing.assert_close(permuted.blocks[0], feature.blocks[0])
            torch.testing.assert_close(permuted.blocks[1], feature.blocks[1].index_select(2, index))
            torch.testing.assert_close(permuted.blocks[2], feature.blocks[2].index_select(2, index).index_select(3, index))
            torch.testing.assert_close(
                permuted.blocks[3],
                feature.blocks[3].index_select(2, index).index_select(3, index).index_select(4, index),
            )


def test_real_feature_group_action_all_small_permutations() -> None:
    for n_particles in range(1, 6):
        feature = RealFeature(
            [
                zero_block(batch_size=2, dtype=torch.float64),
                torch.randn(2, 3, n_particles, dtype=torch.float64),
                torch.randn(2, 3, n_particles, n_particles, dtype=torch.float64),
            ]
        )
        all_permutations = [Permutation(tuple(image)) for image in permutations(range(n_particles))]
        for first in all_permutations:
            for second in all_permutations:
                sequential = feature.permute(first).permute(second)
                composed = feature.permute(second.compose(first))
                torch.testing.assert_close(sequential.blocks[2], composed.blocks[2])


def test_real_feature_random_larger_permutations() -> None:
    generator = torch.Generator().manual_seed(5678)
    n_particles = 8
    feature = RealFeature(
        [
            zero_block(batch_size=2, dtype=torch.float64),
            torch.randn(2, 3, n_particles, generator=generator, dtype=torch.float64),
            torch.randn(2, 3, n_particles, n_particles, generator=generator, dtype=torch.float64),
        ]
    )
    for _ in range(25):
        permutation = Permutation(tuple(torch.randperm(n_particles, generator=generator).tolist()))
        index = torch.tensor(permutation.inverse().image)
        permuted = feature.permute(permutation)
        torch.testing.assert_close(permuted.blocks[1], feature.blocks[1].index_select(2, index))
        torch.testing.assert_close(permuted.blocks[2], feature.blocks[2].index_select(2, index).index_select(3, index))
