"""Tests for generic equivariant state contracts."""

from __future__ import annotations

from itertools import permutations

import torch

from spenn.data import ConcatenatedState, EquivariantState, Permutation, RealFeature, RealUpdate, WavefunctionOutput, zero_block


def test_real_states_implement_equivariant_state_protocol() -> None:
    feature = RealFeature([zero_block(dtype=torch.float64), torch.arange(6, dtype=torch.float64).reshape(1, 2, 3)])
    update = RealUpdate([zero_block(dtype=torch.float64), torch.ones(1, 2, 3, dtype=torch.float64)])

    assert isinstance(feature, EquivariantState)
    assert isinstance(update, EquivariantState)


def test_concatenated_state_permute_applies_componentwise() -> None:
    feature = RealFeature([zero_block(dtype=torch.float64), torch.arange(6, dtype=torch.float64).reshape(1, 2, 3)])
    update = RealUpdate([zero_block(dtype=torch.float64), torch.ones(1, 2, 3, dtype=torch.float64)])
    state = ConcatenatedState((feature, update))
    permutation = Permutation((2, 0, 1))

    permuted = state.permute(permutation)

    assert isinstance(permuted[0], RealFeature)
    assert isinstance(permuted[1], RealUpdate)
    torch.testing.assert_close(permuted[0].blocks[1], feature.permute(permutation).blocks[1])
    torch.testing.assert_close(permuted[1].blocks[1], update.permute(permutation).blocks[1])


def test_concatenated_state_permute_is_exhaustive_for_small_particle_counts() -> None:
    for n_particles in range(1, 6):
        feature = RealFeature(
            [
                zero_block(dtype=torch.float64),
                torch.arange(2 * n_particles, dtype=torch.float64).reshape(1, 2, n_particles),
            ]
        )
        update = RealUpdate([zero_block(dtype=torch.float64), -feature.blocks[1]])
        state = ConcatenatedState((feature, update))
        for image in permutations(range(n_particles)):
            permutation = Permutation(tuple(image))
            permuted = state.permute(permutation)
            torch.testing.assert_close(permuted[0].blocks[1], feature.permute(permutation).blocks[1])
            torch.testing.assert_close(permuted[1].blocks[1], update.permute(permutation).blocks[1])


def test_concatenated_state_permute_handles_random_larger_permutations() -> None:
    generator = torch.Generator().manual_seed(1234)
    n_particles = 8
    feature = RealFeature(
        [
            zero_block(batch_size=2, dtype=torch.float64),
            torch.randn(2, 3, n_particles, generator=generator, dtype=torch.float64),
        ]
    )
    update = RealUpdate(
        [
            zero_block(batch_size=2, dtype=torch.float64),
            torch.randn(2, 3, n_particles, generator=generator, dtype=torch.float64),
        ]
    )
    state = ConcatenatedState((feature, update))
    for _ in range(20):
        permutation = Permutation(tuple(torch.randperm(n_particles, generator=generator).tolist()))
        permuted = state.permute(permutation)
        torch.testing.assert_close(permuted[0].blocks[1], feature.permute(permutation).blocks[1])
        torch.testing.assert_close(permuted[1].blocks[1], update.permute(permutation).blocks[1])


def test_wavefunction_output_has_identity_permutation_action() -> None:
    output = WavefunctionOutput(logabs=torch.zeros(2), sign=torch.ones(2))

    permuted = output.permute(Permutation((1, 0)))

    assert permuted is not output
    torch.testing.assert_close(permuted.logabs, output.logabs)
    torch.testing.assert_close(permuted.sign, output.sign)


def test_wavefunction_output_identity_action_for_all_small_permutations() -> None:
    output = WavefunctionOutput(logabs=torch.arange(4, dtype=torch.float64), sign=torch.ones(4, dtype=torch.float64))
    for n_particles in range(1, 6):
        for image in permutations(range(n_particles)):
            permuted = output.permute(Permutation(tuple(image)))
            torch.testing.assert_close(permuted.logabs, output.logabs)
            torch.testing.assert_close(permuted.sign, output.sign)
