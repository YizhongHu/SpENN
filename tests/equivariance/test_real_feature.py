"""Tests for RealFeature permutation actions."""

from __future__ import annotations

import torch

from spenn.data.indices import permute_tuple_axes
from spenn.data.permutation import Permutation, all_permutations
from spenn.data.real import RealFeature, zero_block


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

    torch.testing.assert_close(
        permuted.blocks[1],
        permute_tuple_axes(feature.blocks[1], permutation, axis_start=2, order=1),
    )
    torch.testing.assert_close(
        permuted.blocks[2],
        permute_tuple_axes(feature.blocks[2], permutation, axis_start=2, order=2),
    )


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
        for permutation in all_permutations(n_particles):
            permuted = feature.permute(permutation)
            torch.testing.assert_close(permuted.blocks[0], feature.blocks[0])
            torch.testing.assert_close(
                permuted.blocks[1],
                permute_tuple_axes(feature.blocks[1], permutation, axis_start=2, order=1),
            )
            torch.testing.assert_close(
                permuted.blocks[2],
                permute_tuple_axes(feature.blocks[2], permutation, axis_start=2, order=2),
            )
            torch.testing.assert_close(
                permuted.blocks[3],
                permute_tuple_axes(feature.blocks[3], permutation, axis_start=2, order=3),
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
        permutation_group = all_permutations(n_particles)
        for first in permutation_group:
            for second in permutation_group:
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
        permuted = feature.permute(permutation)
        torch.testing.assert_close(
            permuted.blocks[1],
            permute_tuple_axes(feature.blocks[1], permutation, axis_start=2, order=1),
        )
        torch.testing.assert_close(
            permuted.blocks[2],
            permute_tuple_axes(feature.blocks[2], permutation, axis_start=2, order=2),
        )
