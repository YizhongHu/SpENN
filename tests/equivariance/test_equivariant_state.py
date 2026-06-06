"""Tests for generic equivariant state contracts."""

from __future__ import annotations

import torch

from spenn.data.batch import WavefunctionOutput
from spenn.data.equivariant_state import ConcatenatedState, EquivariantState
from spenn.data.permutation import Permutation, all_permutations
from spenn.data.real import RealFeature, RealUpdate, zero_block


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
        for permutation in all_permutations(n_particles):
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


def test_wavefunction_output_sign_action_preserves_logabs_and_phase_for_varied_shapes() -> None:
    outputs = [
        WavefunctionOutput(logabs=torch.zeros(2), sign=torch.ones(2), aux={"case": "vector"}),
        WavefunctionOutput(
            logabs=torch.randn(2, 3, dtype=torch.float64),
            sign=torch.ones(2, 3, dtype=torch.float64),
            phase=torch.randn(2, 3, dtype=torch.float64),
            aux={"case": "matrix"},
        ),
    ]

    for output in outputs:
        for permutation in (Permutation((1, 0)), Permutation((2, 0, 1)), Permutation((3, 1, 0, 2))):
            permuted = output.permute(permutation)
            assert permuted is not output
            torch.testing.assert_close(permuted.logabs, output.logabs)
            torch.testing.assert_close(permuted.sign, output.sign * permutation.sign)
            if output.phase is None:
                assert permuted.phase is None
            else:
                torch.testing.assert_close(permuted.phase, output.phase)
            assert permuted.aux == output.aux


def test_wavefunction_output_sign_action_for_all_small_permutations() -> None:
    output = WavefunctionOutput(logabs=torch.arange(4, dtype=torch.float64), sign=torch.ones(4, dtype=torch.float64))
    for n_particles in range(1, 6):
        for permutation in all_permutations(n_particles):
            permuted = output.permute(permutation)
            torch.testing.assert_close(permuted.logabs, output.logabs)
            torch.testing.assert_close(permuted.sign, output.sign * permutation.sign)


def test_wavefunction_output_sign_action_for_random_larger_permutations_with_phase() -> None:
    generator = torch.Generator().manual_seed(8765)
    output = WavefunctionOutput(
        logabs=torch.randn(2, 3, generator=generator, dtype=torch.float64),
        sign=torch.ones(2, 3, dtype=torch.float64),
        phase=torch.randn(2, 3, generator=generator, dtype=torch.float64),
        aux={"tag": "phase"},
    )
    for n_particles in (7, 11):
        for _ in range(20):
            permutation = Permutation(tuple(torch.randperm(n_particles, generator=generator).tolist()))
            permuted = output.permute(permutation)
            torch.testing.assert_close(permuted.logabs, output.logabs)
            torch.testing.assert_close(permuted.sign, output.sign * permutation.sign)
            assert permuted.phase is not None
            torch.testing.assert_close(permuted.phase, output.phase)
            assert permuted.aux == output.aux
