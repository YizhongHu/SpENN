"""Tests for readout scaffold trainability."""

from __future__ import annotations

import pytest
import torch
from typeguard import TypeCheckError

from tpen.data.batch import ElectronBatch
from tpen.data.permutation import all_permutations
from tpen.data.real import Feature, zero_block
from tpen.nn.readout import PfaffianReadout
from tpen.nn.readout.pfaffian import _ODD_PADDING_IRREP, pfaffian


class BlockContainer:
    def __init__(self, blocks: list[torch.Tensor]) -> None:
        self.blocks = blocks

    def __contains__(self, order: int) -> bool:
        return 0 <= order < len(self.blocks)


def _batch(n_electrons: int = 2) -> ElectronBatch:
    return ElectronBatch(positions=torch.zeros(1, n_electrons, 1, dtype=torch.float64))


def _pfaffian_features(n_electrons: int = 2) -> Feature:
    pair = torch.zeros(1, 2, n_electrons, n_electrons, dtype=torch.float64)
    pair[:, :, 0, 1] = torch.tensor([2.0, 4.0], dtype=torch.float64)
    pair[:, :, 1, 0] = -pair[:, :, 0, 1]
    one_body = torch.zeros(1, 1, n_electrons, dtype=torch.float64)
    return Feature([zero_block(dtype=torch.float64), one_body, pair])


def test_pfaffian_matches_known_four_by_four_formula() -> None:
    matrix = torch.tensor(
        [
            [0.0, 2.0, 3.0, 7.0],
            [-2.0, 0.0, 5.0, 11.0],
            [-3.0, -5.0, 0.0, 13.0],
            [-7.0, -11.0, -13.0, 0.0],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        pfaffian(matrix),
        torch.tensor(2.0 * 13.0 - 3.0 * 11.0 + 7.0 * 5.0, dtype=torch.float64),
    )


def test_pfaffian_readout_weights_are_fixed_by_default() -> None:
    readout = PfaffianReadout(pair_channels=2)

    output = readout(_pfaffian_features(), _batch())

    assert output.logabs.shape == (1,)
    assert "channel_weights" not in dict(readout.named_parameters())
    assert "channel_weight_buffer" in dict(readout.named_buffers())
    assert "border_weight_buffer" not in dict(readout.named_buffers())


def test_pfaffian_readout_trainable_flag_registers_weights() -> None:
    readout = PfaffianReadout(pair_channels=2, trainable=True)

    readout(_pfaffian_features(), _batch())

    parameters = dict(readout.named_parameters())
    assert set(parameters) == {"channel_weights"}
    assert parameters["channel_weights"].requires_grad


def test_pfaffian_readout_is_antisymmetric_under_even_particle_permutations() -> None:
    pair = torch.tensor(
        [
            [
                [
                    [0.0, 2.0, 3.0, 7.0],
                    [-2.0, 0.0, 5.0, 11.0],
                    [-3.0, -5.0, 0.0, 13.0],
                    [-7.0, -11.0, -13.0, 0.0],
                ]
            ]
        ],
        dtype=torch.float64,
    )
    features = Feature([zero_block(dtype=torch.float64), torch.empty(1, 0, 4, dtype=torch.float64), pair])
    batch = _batch(n_electrons=4)
    readout = PfaffianReadout(channels=1)

    output = readout(features, batch)
    for permutation in all_permutations(4):
        permuted_output = readout(features.permute(permutation), batch.permute(permutation))

        torch.testing.assert_close(permuted_output.logabs, output.logabs)
        torch.testing.assert_close(permuted_output.sign, output.sign * permutation.sign)


def test_pfaffian_readout_uses_one_irrep_padding_block_for_odd_electrons() -> None:
    one_body = torch.tensor([[[7.0, 11.0, 13.0]]], dtype=torch.float64)
    pair = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    pair[:, :, 0, 1] = 2.0
    pair[:, :, 0, 2] = 3.0
    pair[:, :, 1, 2] = 5.0
    pair = pair - pair.transpose(-1, -2)
    features = Feature([zero_block(dtype=torch.float64), one_body, pair])

    assert _ODD_PADDING_IRREP.parts == (1,)
    output = PfaffianReadout(channels=1)(features, _batch(n_electrons=3))

    expected = torch.tensor([2.0 * 13.0 - 3.0 * 11.0 + 7.0 * 5.0], dtype=torch.float64)
    torch.testing.assert_close(output.aux["pfaffian"], expected)
    torch.testing.assert_close(output.logabs, expected.log())
    torch.testing.assert_close(output.sign, torch.ones_like(expected))
    torch.testing.assert_close(output.aux["K"] + output.aux["K"].transpose(-1, -2), torch.zeros_like(output.aux["K"]))


def test_pfaffian_readout_is_antisymmetric_under_odd_particle_permutations() -> None:
    one_body = torch.tensor([[[7.0, 11.0, 13.0]]], dtype=torch.float64)
    pair = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    pair[:, :, 0, 1] = 2.0
    pair[:, :, 0, 2] = 3.0
    pair[:, :, 1, 2] = 5.0
    pair = pair - pair.transpose(-1, -2)
    features = Feature([zero_block(dtype=torch.float64), one_body, pair])
    batch = _batch(n_electrons=3)
    readout = PfaffianReadout(channels=1)

    output = readout(features, batch)
    for permutation in all_permutations(3):
        permuted_output = readout(features.permute(permutation), batch.permute(permutation))

        torch.testing.assert_close(permuted_output.logabs, output.logabs)
        torch.testing.assert_close(permuted_output.sign, output.sign * permutation.sign)


def test_pfaffian_readout_builds_per_channel_bordered_kernels() -> None:
    # B1: each channel keeps its own skew kernel; odd-n padding borders every
    # channel with that channel's order-1 block (no channel mixing in the
    # kernel).
    one_body = torch.tensor([[[2.0, 4.0, 6.0], [10.0, 14.0, 18.0]]], dtype=torch.float64)
    pair = torch.zeros(1, 2, 3, 3, dtype=torch.float64)
    pair[:, 0, 0, 1] = 2.0
    pair[:, 0, 0, 2] = 3.0
    pair[:, 0, 1, 2] = 5.0
    pair[:, 1, 0, 1] = 4.0
    pair[:, 1, 0, 2] = 9.0
    pair[:, 1, 1, 2] = 11.0
    pair = pair - pair.transpose(-1, -2)
    features = Feature([zero_block(dtype=torch.float64), one_body, pair])

    kernel = PfaffianReadout(channels=2).build_skew_kernel(features, _batch(n_electrons=3))

    expected = torch.zeros(1, 2, 4, 4, dtype=torch.float64)
    expected[:, :, :-1, :-1] = pair
    expected[:, :, :-1, -1] = one_body
    expected[:, :, -1, :-1] = -one_body
    torch.testing.assert_close(kernel, expected)


def test_pfaffian_readout_rejects_per_channel_padding_channel_mismatch() -> None:
    # Per-channel odd-n padding is only defined when the order-1 block
    # carries one border per pair channel.
    one_body = torch.tensor([[[2.0, 4.0, 6.0]]], dtype=torch.float64)
    pair = torch.zeros(1, 2, 3, 3, dtype=torch.float64)
    features = Feature([zero_block(dtype=torch.float64), one_body, pair])

    with pytest.raises(ValueError, match="match pair channels"):
        PfaffianReadout(channels=2)(features, _batch(n_electrons=3))


def test_pfaffian_readout_requires_one_irrep_padding_for_odd_electron_systems() -> None:
    pair = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    features_without_border = Feature([zero_block(dtype=torch.float64), torch.empty(1, 0, 3, dtype=torch.float64), pair])

    with pytest.raises(KeyError, match=r"irrep \(1\)"):
        PfaffianReadout(channels=1)(features_without_border, _batch(n_electrons=3))


def test_pfaffian_readout_does_not_expose_odd_padding_toggle() -> None:
    with pytest.raises(TypeError, match="allow_odd_electron_bordered"):
        PfaffianReadout(allow_odd_electron_bordered=False, channels=1)  # type: ignore[call-arg]


def test_pfaffian_readout_does_not_own_harmonic_confinement() -> None:
    with pytest.raises(TypeError, match="envelope"):
        PfaffianReadout(envelope=object(), channels=1)  # type: ignore[call-arg]


def test_pfaffian_readout_rejects_malformed_kernel_inputs() -> None:
    readout = PfaffianReadout(pair_channels=2)
    malformed_pair = BlockContainer([zero_block(dtype=torch.float64), torch.empty(1, 0, 3), torch.zeros(1, 1, 3)])
    feature = _pfaffian_features()
    border_mismatch = BlockContainer(
        [
            zero_block(dtype=torch.float64),
            torch.ones(1, 1, 4, dtype=torch.float64),
            torch.zeros(1, 1, 3, 3, dtype=torch.float64),
        ]
    )

    with pytest.raises(TypeCheckError, match="Feature"):
        readout.build_skew_kernel(malformed_pair)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Feature batch size"):
        readout.build_skew_kernel(feature, ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64)))
    with pytest.raises(TypeCheckError, match="Feature"):
        readout.build_skew_kernel(border_mismatch)  # type: ignore[arg-type]


def test_pfaffian_readout_returns_empty_pfaffian_for_zero_electrons() -> None:
    features = Feature(
        [
            zero_block(batch_size=1, dtype=torch.float64),
            torch.empty(1, 1, 0, dtype=torch.float64),
            torch.empty(1, 2, 0, 0, dtype=torch.float64),
        ]
    )
    batch = _batch(n_electrons=0)

    output = PfaffianReadout(pair_channels=2)(features, batch)

    torch.testing.assert_close(output.logabs, torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(output.sign, torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(output.aux["pfaffian"], torch.ones(1, dtype=torch.float64))


def _two_channel_distinguishing_features() -> tuple[Feature, torch.Tensor, torch.Tensor]:
    """T6 (B1) two-channel 4x4 case: per-channel Pfaffians with hand values."""

    def skew_from_upper(a12, a13, a14, a23, a24, a34):
        matrix = torch.zeros(4, 4, dtype=torch.float64)
        matrix[0, 1], matrix[0, 2], matrix[0, 3] = a12, a13, a14
        matrix[1, 2], matrix[1, 3], matrix[2, 3] = a23, a24, a34
        return matrix - matrix.T

    # Pf[skew] = a12*a34 - a13*a24 + a14*a23.
    kernel_a = skew_from_upper(2.0, 3.0, 7.0, 5.0, 11.0, 13.0)
    kernel_b = skew_from_upper(1.0, -4.0, 2.0, 6.0, 3.0, -5.0)
    pf_a = torch.tensor(2.0 * 13.0 - 3.0 * 11.0 + 7.0 * 5.0, dtype=torch.float64)
    pf_b = torch.tensor(1.0 * -5.0 - -4.0 * 3.0 + 2.0 * 6.0, dtype=torch.float64)
    pair = torch.stack([kernel_a, kernel_b]).unsqueeze(0)
    features = Feature([zero_block(dtype=torch.float64), torch.empty(1, 0, 4, dtype=torch.float64), pair])
    return features, pf_a, pf_b


def test_pfaffian_readout_matches_per_channel_reference_and_differs_from_mixed_kernel() -> None:
    # T6 channel-semantics case (B1): the module must compute
    # Psi = sum_c w_c Pf[skew_c] and NOT Pf[sum_c w_c skew_c]. Non-uniform
    # weights make the two function classes numerically distinct here.
    from tests.helpers.tpen_reference import mixed_kernel_pfaffian_readout, per_channel_pfaffian_readout

    features, pf_a, pf_b = _two_channel_distinguishing_features()
    weights = torch.tensor([0.75, 0.25], dtype=torch.float64)
    readout = PfaffianReadout(channels=2, trainable=True).to(dtype=torch.float64)
    with torch.no_grad():
        readout.channel_weights.copy_(weights)

    output = readout(features, _batch(n_electrons=4))

    expected_psi = (weights[0] * pf_a + weights[1] * pf_b).reshape(1)
    reference_psi = per_channel_pfaffian_readout(features, weights)
    rejected_psi = mixed_kernel_pfaffian_readout(features, weights)
    torch.testing.assert_close(output.aux["pfaffian"], expected_psi)
    torch.testing.assert_close(output.aux["pfaffian"], reference_psi)
    torch.testing.assert_close(output.logabs, expected_psi.abs().log())
    torch.testing.assert_close(output.sign, torch.sign(expected_psi))
    torch.testing.assert_close(
        output.aux["channel_pfaffians"], torch.stack([pf_a, pf_b]).reshape(1, 2)
    )
    assert not torch.allclose(output.aux["pfaffian"], rejected_psi)


def test_pfaffian_readout_stays_antisymmetric_with_distinct_channel_weights() -> None:
    # T2/T6: per-channel readout must keep sign equivariance even when the
    # channel weights are non-uniform (each Pf flips with det(P)).
    features, _pf_a, _pf_b = _two_channel_distinguishing_features()
    batch = _batch(n_electrons=4)
    readout = PfaffianReadout(channels=2, trainable=True).to(dtype=torch.float64)
    with torch.no_grad():
        readout.channel_weights.copy_(torch.tensor([0.75, 0.25], dtype=torch.float64))

    output = readout(features, batch)
    for permutation in all_permutations(4):
        permuted_output = readout(features.permute(permutation), batch.permute(permutation))

        torch.testing.assert_close(permuted_output.logabs, output.logabs)
        torch.testing.assert_close(permuted_output.sign, output.sign * permutation.sign)


def test_pfaffian_readout_gradients_reach_channel_weights() -> None:
    # T12: backward through logabs must reach the readout weights with a
    # finite, not-identically-zero gradient.
    features, _pf_a, _pf_b = _two_channel_distinguishing_features()
    readout = PfaffianReadout(channels=2, trainable=True).to(dtype=torch.float64)

    readout(features, _batch(n_electrons=4)).logabs.sum().backward()

    gradient = readout.channel_weights.grad
    assert gradient is not None
    assert torch.all(torch.isfinite(gradient))
    assert gradient.abs().sum() > 0
