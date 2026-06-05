"""Tests for RealUpdate permutation actions."""

from __future__ import annotations

from itertools import permutations

import torch

from spenn.data import Permutation, RealUpdate, zero_block


def _update() -> RealUpdate:
    return RealUpdate(
        [
            zero_block(dtype=torch.float64),
            torch.arange(1 * 2 * 3, dtype=torch.float64).reshape(1, 2, 3),
            torch.arange(1 * 2 * 3 * 3, dtype=torch.float64).reshape(1, 2, 3, 3),
        ]
    )


def test_real_update_identity_and_composition() -> None:
    update = _update()
    first = Permutation((1, 0, 2))
    second = Permutation((2, 1, 0))

    identity = update.permute(Permutation.identity(3))
    sequential = update.permute(first).permute(second)
    composed = update.permute(second.compose(first))

    torch.testing.assert_close(identity.blocks, update.blocks)
    torch.testing.assert_close(sequential.blocks, composed.blocks)


def test_real_update_all_small_permutations_and_orders() -> None:
    for n_particles in range(1, 6):
        update = RealUpdate(
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
            index = torch.tensor(permutation.inverse().image)
            permuted = update.permute(permutation)
            torch.testing.assert_close(permuted.blocks[1], update.blocks[1].index_select(2, index))
            torch.testing.assert_close(permuted.blocks[2], update.blocks[2].index_select(2, index).index_select(3, index))
            torch.testing.assert_close(
                permuted.blocks[3],
                update.blocks[3].index_select(2, index).index_select(3, index).index_select(4, index),
            )


def test_real_update_random_larger_permutations() -> None:
    generator = torch.Generator().manual_seed(91011)
    n_particles = 9
    update = RealUpdate(
        [
            zero_block(batch_size=2, dtype=torch.float64),
            torch.randn(2, 3, n_particles, generator=generator, dtype=torch.float64),
            torch.randn(2, 3, n_particles, n_particles, generator=generator, dtype=torch.float64),
        ]
    )
    for _ in range(25):
        permutation = Permutation(tuple(torch.randperm(n_particles, generator=generator).tolist()))
        index = torch.tensor(permutation.inverse().image)
        permuted = update.permute(permutation)
        torch.testing.assert_close(permuted.blocks[1], update.blocks[1].index_select(2, index))
        torch.testing.assert_close(permuted.blocks[2], update.blocks[2].index_select(2, index).index_select(3, index))
